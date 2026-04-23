import os

from datasets import load_dataset
from transformers import AutoTokenizer

from config import TOKENIZER_ID


class PubMedQADataset:
    def __init__(self, tokenizer_id: str = TOKENIZER_ID):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.train_data = load_dataset("pubmed_qa", "pqa_artificial", split="train")
        self.test_data = load_dataset("pubmed_qa", "pqa_labeled", split="train")

        self.label_map = {"yes": 0, "no": 1, "maybe": 2}

    def _preprocess_med(self, sample, rlvr: bool = False):
        """Internal mapping function to structure the conversation."""
        context_text = " ".join(sample["context"]["contexts"])

        # Define the conversation structure
        messages = [
            {
                "role": "user",
                "content": (
                    f"Context: {context_text}\n\n"
                    f"Question: {sample['question']}\n\n"
                    "Answer the question with reasoning. "
                    "End your response with 'Final Decision: [yes/no/maybe]'."
                ),
            },
            {
                "role": "assistant",
                "content": f"{sample['long_answer']}\nFinal Decision: {sample['final_decision']}",
            },
        ]

        full_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        res = {
            "text": full_prompt,  # Used by SFTTrainer
            "messages": messages,  # Used for evaluation
            "ground_truth": sample["final_decision"],
        }

        if rlvr:
            res["prompt"] = self.tokenizer.apply_chat_template(
                [messages[0]], tokenize=False, add_generation_prompt=True
            )

        return res

    def get_dataset(
        self,
        n_sft_samples="all",
        n_rlvr_samples="all",
        n_test_samples="all",
        rlvr: bool = False,
    ):
        """Splits and returns the datasets based on user requirements."""

        # Shuffle and split training data
        train_shuffled = self.train_data.shuffle(seed=123)
        split_dict = train_shuffled.train_test_split(test_size=0.5, shuffle=False)

        sft_raw = split_dict["train"]
        rlvr_raw = split_dict["test"]
        test_raw = self.test_data

        # Slicing logic
        def slice_data(dataset, n):
            if n != "all":
                return dataset.select(range(min(n, len(dataset))))
            return dataset

        sft_raw = slice_data(sft_raw, n_sft_samples)
        rlvr_raw = slice_data(rlvr_raw, n_rlvr_samples)
        test_raw = slice_data(test_raw, n_test_samples)

        # Apply mapping
        sft_dataset = sft_raw.map(
            lambda x: self._preprocess_med(x, rlvr=False),
            remove_columns=self.train_data.column_names,
            num_proc=os.cpu_count(),
        )

        rlvr_dataset = rlvr_raw.map(
            lambda x: self._preprocess_med(x, rlvr=True),
            remove_columns=self.train_data.column_names,
            num_proc=os.cpu_count(),
        )

        test_dataset = test_raw.map(
            lambda x: self._preprocess_med(x, rlvr=rlvr),
            remove_columns=self.test_data.column_names,
            num_proc=os.cpu_count(),
        )

        return sft_dataset, rlvr_dataset, test_dataset

    @staticmethod
    def get_decision(text: str) -> str:
        """Static method to extract decision from model text."""
        text = text.lower().strip()
        if "final decision: yes" in text:
            return "yes"
        if "final decision: no" in text:
            return "no"
        if "final decision: maybe" in text:
            return "maybe"
        return "unknown"
