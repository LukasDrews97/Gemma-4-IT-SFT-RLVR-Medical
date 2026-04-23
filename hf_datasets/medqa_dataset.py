import os

from datasets import load_dataset
from transformers import AutoTokenizer

from config import TOKENIZER_ID


def get_dataset(n_test_samples="all"):
    # Using the standardized 4-option version
    dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    tokenizer.pad_token = tokenizer.eos_token

    def preprocess_medqa(sample):
        # GBaker format uses a dictionary for options: {'A': 'text', 'B': 'text', ...}
        opts = sample["options"]
        options_formatted = "\n".join([f"{k}) {v}" for k, v in opts.items()])

        # In this dataset, answer_idx is usually already the letter (A, B, C, or D)
        ground_truth_letter = str(sample["answer_idx"]).upper()

        messages = [
            {
                "role": "user",
                "content": (
                    f"Question: {sample['question']}\n\n"
                    f"Options:\n{options_formatted}\n\n"
                    "Answer the question with reasoning. "
                    "End your response with 'Final Decision: [A/B/C/D]'."
                ),
            }
        ]

        full_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        return {
            "text": full_prompt,
            "messages": messages,
            "ground_truth": ground_truth_letter,
        }

    if n_test_samples != "all":
        n_test_samples = min(n_test_samples, len(dataset))
        dataset = dataset.select(range(n_test_samples))

    test_dataset = dataset.map(
        preprocess_medqa, remove_columns=dataset.column_names, num_proc=os.cpu_count()
    )

    return test_dataset


def get_decision(text: str) -> str:
    """Standardized parser for final decisions."""
    # We look for the exact string pattern from SFT
    text = text.lower().strip()
    for letter in ["a", "b", "c", "d"]:
        if f"final decision: {letter}" in text:
            return letter
    return ""
