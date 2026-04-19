import os

from datasets import load_dataset
from transformers import AutoTokenizer

from config import TOKENIZER_ID


def get_dataset(n_train_samples="all", n_test_samples="all"):
    train_data = load_dataset("pubmed_qa", "pqa_artificial", split="train")
    test_data = load_dataset("pubmed_qa", "pqa_labeled", split="train")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)

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
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        return {
            "text": full_prompt,              # Used by SFTTrainer
            "messages": messages,             # Used for evaluation
            "ground_truth": sample["final_decision"],
        }

    train_dataset = train_data.shuffle(seed=123)
    test_dataset = test_data

    if n_train_samples != "all":
        n_train_samples = min(n_train_samples, len(train_dataset))
        train_dataset = train_dataset.select(range(n_train_samples))

    if n_test_samples != "all":
        n_test_samples = min(n_test_samples, len(test_dataset))
        test_dataset = test_dataset.select(range(n_test_samples))

    train_dataset = train_dataset.map(
        preprocess_med, remove_columns=train_data.column_names, num_proc=os.cpu_count()
    )
    test_dataset = test_dataset.map(
        preprocess_med, remove_columns=test_data.column_names, num_proc=os.cpu_count()
    )

    return train_dataset, test_dataset
