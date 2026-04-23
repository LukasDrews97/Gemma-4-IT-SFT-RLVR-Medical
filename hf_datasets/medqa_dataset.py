import os

from datasets import load_dataset
from transformers import AutoTokenizer

from config import TOKENIZER_ID

'''
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
'''


class MedQADataset:
    def __init__(self, tokenizer_id: str = TOKENIZER_ID):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load the 4-option version of MedQA
        self.raw_dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")

        self.label_map = {0: "A", 1: "B", 2: "C", 3: "D"}

    def _preprocess_medqa(self, sample):
        """Internal mapping function to format the USMLE questions."""
        # Format options: {'A': 'text', 'B': 'text'} -> "A) text\nB) text"
        opts = sample["options"]
        options_formatted = "\n".join([f"{k}) {v}" for k, v in opts.items()])

        # Ensure ground truth is a capitalized letter (A, B, C, or D)
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

        full_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        return {
            "text": full_prompt,  # For SFT
            "messages": messages,  # For Evaluation
            "ground_truth": ground_truth_letter,
        }

    def get_dataset(self, n_test_samples="all"):
        """Returns the processed test dataset."""
        dataset = self.raw_dataset

        if n_test_samples != "all":
            n_test_samples = min(n_test_samples, len(dataset))
            dataset = dataset.select(range(n_test_samples))

        test_dataset = dataset.map(
            self._preprocess_medqa,
            remove_columns=dataset.column_names,
            num_proc=os.cpu_count(),
        )

        return None, None, test_dataset

    @staticmethod
    def get_decision(text: str) -> str:
        """
        Extracts the multiple choice letter from the model output.
        Returns the lowercase letter (a, b, c, or d) or an empty string.
        """
        text = text.lower().strip()
        for letter in ["a", "b", "c", "d"]:
            # Check for the specific pattern defined in the prompt
            if f"final decision: {letter}" in text:
                return letter
        return ""
