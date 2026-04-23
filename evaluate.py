from pathlib import Path

import torch
from llama_cpp import Llama
from peft import PeftModel
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall
from tqdm import tqdm
from transformers import AutoModelForCausalLM, DataCollatorWithPadding

from config import (
    MODEL_ID,
    N_TEST_SAMPLES,
    QUANTIZATION_FOLDER,
    RLVR_PATH,
    SFT_PATH,
)
from hf_datasets.medmcqa_dataset import MedMCQADataset
from hf_datasets.medqa_dataset import MedQADataset
from hf_datasets.pubmedqa_dataset import PubMedQADataset


def main() -> None:
    def get_base_model():
        return AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
            offload_buffers=True,
        )

    def evaluate(
        model, dataset_class, n_test_samples, name="Model", device="cuda", batch_size=32
    ):
        device = model.device if hasattr(model, "device") else device

        print(f"Evaluating {name}")

        precision_metric = MulticlassPrecision(
            num_classes=len(dataset_class.label_map.keys()), average="macro"
        ).to(device)
        recall_metric = MulticlassRecall(
            num_classes=len(dataset_class.label_map.keys()), average="macro"
        ).to(device)

        _, _, dataset = dataset_class.get_dataset(n_test_samples=n_test_samples)

        correct = 0
        total = len(dataset)

        if isinstance(model, Llama):
            for sample in tqdm(dataset):

                prompt = dataset_class.tokenizer.apply_chat_template(
                    sample["messages"][:1], tokenize=False, add_generation_prompt=True
                )

                response = model(
                    prompt=prompt,
                    max_tokens=2048,
                    temperature=0.0,
                    echo=False,
                )
                prediction_text = response["choices"][0]["text"]
                correct += update_metrics(
                    prediction_text,
                    sample["ground_truth"],
                    dataset_class,
                    precision_metric,
                    recall_metric,
                    device,
                )

        else:

            def tokenize_fn(examples):
                prompts = [
                    dataset_class.tokenizer.apply_chat_template(
                        [m[0]], tokenize=False, add_generation_prompt=True
                    )
                    for m in examples["messages"]
                ]
                return dataset_class.tokenizer(
                    prompts, padding=False, truncation=True, max_length=2048
                )

            tokenized_dataset = dataset.map(
                tokenize_fn, batched=True, remove_columns=dataset.column_names
            )
            tokenized_dataset = tokenized_dataset.add_column(
                "ground_truth", dataset["ground_truth"]
            )
            tokenized_dataset.set_format(
                type="torch", columns=["input_ids", "attention_mask"]
            )

            collator = DataCollatorWithPadding(tokenizer=dataset_class.tokenizer)
            dataloader = DataLoader(
                tokenized_dataset, batch_size=batch_size, collate_fn=collator
            )

            model.eval()
            for idx, batch in enumerate(tqdm(dataloader)):
                start = idx * batch_size
                end = start + len(batch["input_ids"])
                ground_truths = dataset["ground_truth"][start:end]
                batch.to(device)

                with torch.no_grad():
                    outputs = model.generate(
                        **batch,
                        max_new_tokens=2048,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=dataset_class.tokenizer.pad_token_id,
                    )

                input_len = batch["input_ids"].shape[1]
                generated_texts = dataset_class.tokenizer.batch_decode(
                    outputs[:, input_len:], skip_special_tokens=True
                )

                for pred_text, gt_str in zip(generated_texts, ground_truths):
                    correct += update_metrics(
                        pred_text,
                        gt_str,
                        dataset_class,
                        precision_metric,
                        recall_metric,
                        device,
                    )

        accuracy = (correct / total) * 100
        precision = precision_metric.compute().item() * 100
        recall = recall_metric.compute().item() * 100
        print(
            f"{name} | Accuracy: {accuracy:.2f}% | Precision: {precision:.2f}% | Recall: {recall:.2f}%"
        )
        return accuracy, precision, recall

    datasets = [
        ("PubMedQA", PubMedQADataset()),
        ("MedQA", MedQADataset()),
        ("MedMCQA", MedMCQADataset()),
    ]

    for dataset_name, dataset in datasets:
        # Evaluate Base Model
        base_acc, base_pre, base_recall = evaluate(
            get_base_model(),
            dataset,
            n_test_samples=N_TEST_SAMPLES,
            name=f"Gemma 4-{dataset_name}",
        )
        torch.cuda.empty_cache()

        # Evaluate SFT Model
        sft_model = PeftModel.from_pretrained(get_base_model(), SFT_PATH)
        sft_model.eval()

        sft_acc, sft_pre, sft_recall = evaluate(
            sft_model,
            dataset,
            n_test_samples=N_TEST_SAMPLES,
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
            n_test_samples=N_TEST_SAMPLES,
            name=f"Finetuned Gemma 4 (SFT + RLVR)-{dataset_name}",
        )

        del rlvr_model
        torch.cuda.empty_cache()

        # Evaluate quantized models
        gguf_files = list(Path(QUANTIZATION_FOLDER).rglob("*.gguf"))
        # only evaluate sft-rlvr for now
        gguf_files = [f for f in gguf_files if "sft-rlvr" in str(f)]

        for qt_model_path in gguf_files:
            qt_model = Llama(
                model_path=str(qt_model_path),
                n_gpu_layers=-1,
                n_ctx=2048,
                verbose=False,
                flash_attn=True,
            )
            qt_acc, qt_pre, qt_rec = evaluate(
                qt_model,
                dataset,
                n_test_samples=N_TEST_SAMPLES,
                device="cpu",
                name=f"{qt_model_path}-{dataset_name}",
            )
            print(
                f"{qt_model_path}-{dataset_name}: Accuracy: {qt_acc}, Precision: {qt_pre}, Recall: {qt_rec}"
            )
            del qt_model
            torch.cuda.empty_cache()


def update_metrics(pred_text, gt_text, dataset_class, p_metric, r_metric, device):
    """Helper to parse decision and update TorchMetrics."""
    pred_decision = dataset_class.get_decision(pred_text).lower()
    ground_truth = gt_text.lower()

    # Update Precision/Recall tensors
    p_idx = dataset_class.label_map.get(pred_decision, -1)
    g_idx = dataset_class.label_map.get(ground_truth)

    if p_idx != -1 and g_idx is not None:
        p_metric.update(
            torch.tensor([p_idx]).to(device), torch.tensor([g_idx]).to(device)
        )
        r_metric.update(
            torch.tensor([p_idx]).to(device), torch.tensor([g_idx]).to(device)
        )

    return 1 if pred_decision == ground_truth else 0


if __name__ == "__main__":
    main()
