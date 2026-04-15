import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
from pubmedqa_dataset import get_dataset


model_id = "google/gemma-4-E2B"
adapter_path = "./gemma-PubMedQA-SFT"

def main() -> None:
    # 1. Load Tokenizer and Base Model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    _, test_dataset = get_dataset()


    def get_decision(text):
        """Simple parser to extract the 'yes/no/maybe' from the end of the response."""
        text = text.lower().strip()
        if "final decision: yes" in text: return "yes"
        if "final decision: no" in text: return "no"
        if "final decision: maybe" in text: return "maybe"
        return "unknown"

    def evaluate(model, dataset, name="Model"):
        correct = 0
        total = len(dataset)
        
        print(f"Evaluating {name}...")
        for sample in tqdm(dataset):
            prompt = sample["messages"][0]["content"]
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)

            prediction_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            pred_decision = get_decision(prediction_text)

            if pred_decision == sample["ground_truth"].lower():
                correct += 1

        accuracy = (correct / total) * 100
        print(f"{name} Accuracy: {accuracy:.2f}%")
        return accuracy

    # --- STEP 1: Evaluate Base Model ---
    # (Adapters are not loaded yet)
    base_acc = evaluate(model, test_dataset, name="Base Gemma")

    # --- STEP 2: Load Adapter and Evaluate Finetuned Model ---
    print("\nLoading PEFT adapters...")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval() # Set to inference mode

    ft_acc = evaluate(model, test_dataset, name="Finetuned Gemma (SFT)")

    print(f"\nSummary:\nBase: {base_acc}%\nSFT: {ft_acc}%")


if __name__ == "__main__":
    main()
