import os

from datasets import load_dataset
from transformers import AutoTokenizer

from config import TOKENIZER_ID


def get_dataset(
    n_sft_samples="all", n_rlvr_samples="all", n_test_samples="all", rlvr: bool = False
):
    train_data = load_dataset("pubmed_qa", "pqa_artificial", split="train")
    test_data = load_dataset("pubmed_qa", "pqa_labeled", split="train")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    tokenizer.pad_token = tokenizer.eos_token

    # map sample to output
    def preprocess_med(sample):
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

        full_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        res = {
            "text": full_prompt,  # Used by SFTTrainer
            "messages": messages,  # Used for evaluation
            "ground_truth": sample["final_decision"],
        }

        if rlvr:
            res["prompt"] = tokenizer.apply_chat_template(
                [messages[0]], tokenize=False, add_generation_prompt=True
            )

        return res

    train_dataset = train_data.shuffle(seed=123)
    split_dict = train_dataset.train_test_split(test_size=0.5, shuffle=False)
    sft_dataset, rlvr_dataset = split_dict["train"], split_dict["test"]
    test_dataset = test_data

    if n_sft_samples != "all":
        n_sft_samples = min(n_sft_samples, len(sft_dataset))
        sft_dataset = sft_dataset.select(range(n_sft_samples))

    if n_rlvr_samples != "all":
        n_rlvr_samples = min(n_rlvr_samples, len(rlvr_dataset))
        rlvr_dataset = rlvr_dataset.select(range(n_rlvr_samples))

    if n_test_samples != "all":
        n_test_samples = min(n_test_samples, len(test_dataset))
        test_dataset = test_dataset.select(range(n_test_samples))

    sft_dataset = sft_dataset.map(
        preprocess_med, remove_columns=train_data.column_names, num_proc=os.cpu_count()
    )

    rlvr_dataset = rlvr_dataset.map(
        preprocess_med, remove_columns=train_data.column_names, num_proc=os.cpu_count()
    )

    test_dataset = test_dataset.map(
        preprocess_med, remove_columns=test_data.column_names, num_proc=os.cpu_count()
    )

    return sft_dataset, rlvr_dataset, test_dataset


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
