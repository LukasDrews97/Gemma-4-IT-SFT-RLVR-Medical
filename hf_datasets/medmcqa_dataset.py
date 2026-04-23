"""Test-only dataset."""

import os

from datasets import load_dataset
from transformers import AutoTokenizer

from config import TOKENIZER_ID


class MedMCQADataset:
    def __init__(self, tokenizer_id: str = TOKENIZER_ID):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.raw_dataset = load_dataset("openlifescienceai/medmcqa", split="validation")
        self.label_map = {0: "A", 1: "B", 2: "C", 3: "D"}

    def _preprocess_medmcqa(self, sample):
        """Internal mapping function to format MedMCQA questions and options."""
        options = (
            f"A) {sample['opa']}\n"
            f"B) {sample['opb']}\n"
            f"C) {sample['opc']}\n"
            f"D) {sample['opd']}"
        )

        correct_idx = int(sample["cop"])
        ground_truth_letter = self.label_map.get(correct_idx, "Unknown")

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

        full_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        return {
            "text": full_prompt,
            "messages": messages,
            "ground_truth": ground_truth_letter,
            "subject": sample.get("subject_name", "General Medical"),
        }

    def get_dataset(self, n_test_samples="all"):
        """Returns the processed validation dataset."""
        dataset = self.raw_dataset

        if n_test_samples != "all":
            n_test_samples = min(n_test_samples, len(dataset))
            dataset = dataset.select(range(n_test_samples))

        test_dataset = dataset.map(
            self._preprocess_medmcqa,
            remove_columns=dataset.column_names,
            num_proc=os.cpu_count(),
        )

        return None, None, test_dataset

    @staticmethod
    def get_decision(text: str) -> str:
        """Parser for MedMCQA decision extraction."""
        text = text.lower().strip()
        for letter in ["a", "b", "c", "d"]:
            if f"final decision: {letter}" in text:
                return letter
        return ""
