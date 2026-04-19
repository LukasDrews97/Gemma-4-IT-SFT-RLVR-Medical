import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import MODEL_ID, N_TEST_SAMPLES, SFT_PATH, TOKENIZER_ID
from pubmedqa_dataset import get_dataset


def main() -> None:
    # Load Tokenizer and Base Model
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto"
    )

    _, test_dataset = get_dataset(n_test_samples=N_TEST_SAMPLES)

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

    def evaluate(model, dataset, name="Model"):
        correct = 0
        total = len(dataset)

        print(f"Evaluating {name}")
        for sample in tqdm(dataset):
            # 1. Format the prompt correctly using the template
            # We only take the 'user' message and add the generation prompt
            # (this adds the <start_of_turn>model tag)
            prompt = tokenizer.apply_chat_template(
                sample["messages"][:1], tokenize=False, add_generation_prompt=True
            )

            # 2. Tokenize and move to GPU
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            # 3. Generate the response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=2048,  # Enough space for reasoning + decision
                    do_sample=False,  # Greedy decoding for consistent eval
                )

            # 4. SLICE THE OUTPUT TO GET ONLY NEW TOKENS
            # This prevents the context from being part of the prediction string
            input_len = inputs.input_ids.shape[1]
            generated_tokens = outputs[0][input_len:]

            prediction_text = tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            )

            # 5. Parse and Score
            pred_decision = get_decision(prediction_text)

            if pred_decision == sample["ground_truth"].lower():
                correct += 1

            # Optional: Debug print for the first few samples to ensure parsing works
            # if total - len(dataset) < 3:
            #    print(f"\nPred: {pred_decision} | Actual: {sample['ground_truth']}")

        accuracy = (correct / total) * 100
        print(f"{name} Accuracy: {accuracy:.2f}%")
        return accuracy

    # Evaluate Base Model
    base_acc = evaluate(model, test_dataset, name="Gemma 4")

    # Evaluate SFT Model
    model = PeftModel.from_pretrained(model, SFT_PATH)
    model.eval()

    ft_acc = evaluate(model, test_dataset, name="Finetuned Gemma (SFT)")

    print(f"\nSummary:\nBase: {base_acc}%\nSFT: {ft_acc}%")


if __name__ == "__main__":
    main()
