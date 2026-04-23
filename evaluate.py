from pathlib import Path

import torch
from llama_cpp import Llama
from peft import PeftModel
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel

from config import (
    MODEL_ID,
    N_TEST_SAMPLES,
    QUANTIZATION_FOLDER,
    RLVR_PATH,
    SFT_PATH,
    TOKENIZER_ID,
)
from hf_datasets.medmcqa_dataset import get_dataset as medmcqa_get_dataset
from hf_datasets.medmcqa_dataset import get_decision as medmcqa_get_decision
from hf_datasets.medqa_dataset import get_dataset as medqa_get_dataset
from hf_datasets.medqa_dataset import get_decision as medqa_get_decision
from hf_datasets.pubmedqa_dataset import get_dataset as pubmedqa_get_dataset
from hf_datasets.pubmedqa_dataset import get_decision as pubmedqa_get_decision


def main() -> None:
    # Load Tokenizer and Base Model
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)

    def get_base_model():
        return AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
            offload_buffers=True,
        )

    def evaluate(model, dataset, decision_func, label_map, name="Model", device="cuda"):
        device = model.device if hasattr(model, "device") else device
        num_classes = len(label_map.keys())
        #label_map = {"yes": 0, "no": 1, "maybe": 2}

        precision_metric = MulticlassPrecision(
            num_classes=num_classes, average="macro"
        ).to(device)
        recall_metric = MulticlassRecall(num_classes=num_classes, average="macro").to(
            device
        )

        correct = 0
        total = len(dataset)

        print(f"Evaluating {name}")
        for sample in tqdm(dataset):
            prompt = tokenizer.apply_chat_template(
                sample["messages"][:1], tokenize=False, add_generation_prompt=True
            )

            if isinstance(model, Llama):
                response = model(
                    prompt=prompt,
                    max_tokens=2048,
                    temperature=0.0,
                    echo=False,
                )
                prediction_text = response["choices"][0]["text"]
            else:
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                input_len = inputs.input_ids.shape[1]
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=2048,
                        do_sample=False,
                    )
                generated_tokens = outputs[0][input_len:]
                prediction_text = tokenizer.decode(
                    generated_tokens, skip_special_tokens=True
                )

            pred_decision = decision_func(prediction_text)
            ground_truth = sample["ground_truth"].lower()

            if pred_decision == ground_truth:
                correct += 1

            p_idx = label_map.get(pred_decision, -1)
            g_idx = label_map.get(ground_truth)
            if p_idx != -1:
                p_tensor = torch.tensor([p_idx]).to(device)
                g_tensor = torch.tensor([g_idx]).to(device)
                precision_metric.update(p_tensor, g_tensor)
                recall_metric.update(p_tensor, g_tensor)

        accuracy = (correct / total) * 100
        precision = precision_metric.compute().item() * 100
        recall = recall_metric.compute().item() * 100
        print(f"{name} Accuracy: {accuracy:.2f}%")
        print(f"{name} Precision: {precision:.2f}%")
        print(f"{name} Recall: {recall:.2f}%")
        return accuracy, precision, recall

    _, _, pubmedqa = pubmedqa_get_dataset(n_test_samples=N_TEST_SAMPLES)
    medmcqa = medmcqa_get_dataset(n_test_samples=N_TEST_SAMPLES)
    medqa = medqa_get_dataset(n_test_samples=N_TEST_SAMPLES)

    datasets = [
        ("PubMedQA", pubmedqa, pubmedqa_get_decision, {"yes": 0, "no": 1, "maybe": 2}),
        ("MedMCQA", medmcqa, medmcqa_get_decision, {0: "A", 1: "B", 2: "C", 3: "D"}),
        ("MedQA", medqa, medqa_get_decision, {0: "A", 1: "B", 2: "C", 3: "D"}),
    ]

    for dataset_name, dataset, decision_func, label_map in datasets:
        # Evaluate Base Model
        base_acc, base_pre, base_recall = evaluate(
            get_base_model(), dataset, decision_func, label_map, name=f"Gemma 4-{dataset_name}"
        )
        torch.cuda.empty_cache()

        # Evaluate SFT Model
        sft_model = PeftModel.from_pretrained(get_base_model(), SFT_PATH)
        sft_model.eval()

        sft_acc, sft_pre, sft_recall = evaluate(
            sft_model,
            dataset,
            decision_func,
            label_map,
            name=f"Finetuned Gemma 4 (SFT)-{dataset_name}",
        )

        del sft_model
        torch.cuda.empty_cache()

        # Evaluate RLVR Model
        rlvr_model = PeftModel.from_pretrained(get_base_model(), RLVR_PATH)
        rlvr_model.eval()

        rlvr_acc, rlvr_pre, rlvr_recall = evaluate(
            rlvr_model,
            dataset,
            decision_func,
            label_map,
            name=f"Finetuned Gemma 4 (SFT + RLVR)-{dataset_name}",
        )

        del rlvr_model
        torch.cuda.empty_cache()

        # Evaluate quantized models
        gguf_files = list(Path(QUANTIZATION_FOLDER).rglob("*.gguf"))

        for qt_model_path in gguf_files:
            qt_model = Llama(
                model_path=str(qt_model_path),
                n_gpu_layers=-1,
                n_ctx=2048,
                verbose=False,
                flash_attn=True,
            )
            qt_acc, qt_pre, qt_rec = evaluate(
                qt_model, dataset, decision_func, label_map, name=f"{qt_model_path}-{dataset_name}"
            )
            print(
                f"{qt_model_path}-{dataset_name}: Accuracy: {qt_acc}, Precision: {qt_pre}, Recall: {qt_rec}"
            )
            del qt_model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
