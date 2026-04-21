import torch
from peft import PeftModel
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import MODEL_ID, N_TEST_SAMPLES, RLVR_PATH, SFT_PATH, TOKENIZER_ID
from pubmedqa_dataset import get_dataset


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

    _, _, test_dataset = get_dataset(n_test_samples=N_TEST_SAMPLES)

    def evaluate(model, dataset, name="Model"):
        num_classes = 3
        label_map = {"yes": 0, "no": 1, "maybe": 2}

        precision_metric = MulticlassPrecision(
            num_classes=num_classes, average="macro"
        ).to(model.device)
        recall_metric = MulticlassRecall(num_classes=num_classes, average="macro").to(
            model.device
        )

        correct = 0
        total = len(dataset)

        print(f"Evaluating {name}")
        for sample in tqdm(dataset):
            prompt = tokenizer.apply_chat_template(
                sample["messages"][:1], tokenize=False, add_generation_prompt=True
            )

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    do_sample=False,
                )

            # only keep output tokens
            input_len = inputs.input_ids.shape[1]
            generated_tokens = outputs[0][input_len:]

            prediction_text = tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            )

            pred_decision = get_decision(prediction_text)
            ground_truth = sample["ground_truth"].lower()

            if pred_decision == ground_truth:
                correct += 1

            p_idx = label_map.get(pred_decision, -1)
            g_idx = label_map.get(ground_truth)
            if p_idx != -1:
                p_tensor = torch.tensor([p_idx]).to(model.device)
                g_tensor = torch.tensor([g_idx]).to(model.device)
                precision_metric.update(p_tensor, g_tensor)
                recall_metric.update(p_tensor, g_tensor)

        accuracy = (correct / total) * 100
        precision = precision_metric.compute().item() * 100
        recall = recall_metric.compute().item() * 100
        print(f"{name} Accuracy: {accuracy:.2f}%")
        print(f"{name} Precision: {precision:.2f}%")
        print(f"{name} Recall: {recall:.2f}%")
        return accuracy, precision, recall

    # Evaluate Base Model
    base_acc, base_pre, base_recall = evaluate(get_base_model(), test_dataset, name="Gemma 4")
    torch.cuda.empty_cache()

    # Evaluate SFT Model
    sft_model = PeftModel.from_pretrained(get_base_model(), SFT_PATH)
    sft_model.eval()

    sft_acc, sft_pre, sft_recall = evaluate(
        sft_model, test_dataset, name="Finetuned Gemma 4 (SFT)"
    )

    del sft_model
    torch.cuda.empty_cache()

    # Evaluate RLVR Model
    rlvr_model = PeftModel.from_pretrained(get_base_model(), RLVR_PATH)
    rlvr_model.eval()

    rlvr_acc, rlvr_pre, rlvr_recall = evaluate(
        rlvr_model, test_dataset, name="Finetuned Gemma 4 (SFT + RLVR)"
    )

    del rlvr_model
    torch.cuda.empty_cache()

    print("Summary:")
    print(
        f"\nBase:\nAccuracy: {base_acc}, Precision: {base_pre}, Recall: {base_recall}"
    )
    print(f"\nSFT:\nAccuracy: {sft_acc}, Precision: {sft_pre}, Recall: {sft_recall}")
    print(
        f"\nSFT + RLVR:\nAccuracy: {rlvr_acc}, Precision: {rlvr_pre}, Recall: {rlvr_recall}"
    )



def get_decision(text: str) -> str:
    """Simple parser to extract the 'yes/no/maybe' from the end of the response."""
    text = text.lower().strip()
    if "final decision: yes" in text:
        return "yes"
    if "final decision: no" in text:
        return "no"
    if "final decision: maybe" in text:
        return "maybe"
    return "unknown"


if __name__ == "__main__":
    main()
