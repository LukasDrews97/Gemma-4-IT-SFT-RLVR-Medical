from datasets import load_dataset


def get_dataset():
    train_data = load_dataset("pubmed_qa", "pqa_artificial", split="train")
    # Use the expert-labeled set for testing
    test_data = load_dataset("pubmed_qa", "pqa_labeled", split="train")

    # 2. THE FORMATTING FUNCTION
    def preprocess_med(sample):
        # Flatten the context list into a paragraph
        context_text = " ".join(sample["context"]["contexts"])

        # We create a prompt that asks for reasoning first, then the decision
        # This structure is PERFECT for RLVR later.
        user_msg = f"Context: {context_text}\n\nQuestion: {sample['question']}\n\nAnswer the question with reasoning."

        return {
            "messages": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": f"{sample['long_answer']}\nFinal Decision: {sample['final_decision']}"}
            ],
            "ground_truth": sample["final_decision"] # Store this for RLVR rewards later
        }

    # 3. APPLY FORMATTING
    # Select a manageable amount for your 5070 Ti (e.g., 15k-20k)
    train_dataset = train_data.shuffle(seed=42).select(range(40)).map(preprocess_med, remove_columns=train_data.column_names)
    test_dataset = test_data.map(preprocess_med, remove_columns=test_data.column_names)

    return train_dataset, test_dataset
