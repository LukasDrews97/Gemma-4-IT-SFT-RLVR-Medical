"""Test-only dataset."""

import os

from datasets import load_dataset
from transformers import AutoTokenizer

from config import TOKENIZER_ID


def get_dataset(n_test_samples="all"):
    dataset = load_dataset("openlifescienceai/medmcqa", split="validation")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    tokenizer.pad_token = tokenizer.eos_token

    label_map = {0: "A", 1: "B", 2: "C", 3: "D"}

    def preprocess_medmcqa(sample):
        options = (
            f"A) {sample['opa']}\n"
            f"B) {sample['opb']}\n"
            f"C) {sample['opc']}\n"
            f"D) {sample['opd']}"
        )

        # Handle label mapping (ensure it's an int)
        correct_idx = int(sample["cop"])
        ground_truth_letter = label_map.get(correct_idx, "Unknown")

        # Define the conversation structure
        messages = [
            {
                "role": "user",
                "content": (
                    f"Question: {sample['question']}\n\n"
                    f"Options:\n{options}\n\n"
                    "Answer the question with reasoning. "
                    "End your response with 'Final Decision: [A/B/C/D]'."
                ),
            }
        ]

        # Since this is for validation, we only need the user prompt
        # but we include the ground truth for your evaluation script.
        full_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        return {
            "text": full_prompt,  # Prompt for the model
            "messages": messages,
            "ground_truth": ground_truth_letter,
            "subject": sample.get("subject_name", "General Medical"),
        }

    # Selection logic
    if n_test_samples != "all":
        n_test_samples = min(n_test_samples, len(dataset))
        dataset = dataset.select(range(n_test_samples))

    # Map the dataset
    test_dataset = dataset.map(
        preprocess_medmcqa, remove_columns=dataset.column_names, num_proc=os.cpu_count()
    )

    return test_dataset


def get_decision(text: str) -> str:
    """Simple parser to extract the 'a/b/c/d' from the end of the response."""
    text = text.lower().strip()
    if "final decision: a" in text:
        return "a"
    if "final decision: b" in text:
        return "b"
    if "final decision: c" in text:
        return "c"
    if "final decision: d" in text:
        return "d"
    return ""
