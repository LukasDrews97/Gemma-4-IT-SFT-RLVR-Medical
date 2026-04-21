import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from config import (
    MODEL_ID,
    RLVR_PATH,
    SFT_DEPLOYMENT_PATH,
    SFT_PATH,
    SFT_RLVR_DEPLOYMENT_PATH,
    TOKENIZER_ID,
)


def merge_and_export(adapter_path: str, output_path: str) -> None:
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)

    model = PeftModel.from_pretrained(base_model, adapter_path, device_map="cpu")
    merged_model = model.merge_and_unload()
    merged_model.config.tie_word_embeddings = False

    print(f"Saving to {output_path}")
    merged_model.save_pretrained(
        output_path, safe_serialization=True, max_shard_size="2GB"
    )
    tokenizer.save_pretrained(output_path)
    print("Done.\n")


def test_model_deployment(path: str, reasoning: bool) -> None:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    pipe = pipeline(
        "text-generation",
        model=path,
        tokenizer=tokenizer,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    context = (
        "In patients with chronic obstructive pulmonary disease (COPD), "
        "long-acting beta2-agonists (LABAs) are used to improve lung function. "
        "Recent studies investigated whether adding a long-acting muscarinic antagonist (LAMA) "
        "further reduces exacerbation rates compared to LABA monotherapy."
    )

    question = "Does LAMA/LABA combination therapy reduce COPD exacerbations more effectively than LABA alone?"

    content = f"Context: {context}\n\nQuestion: {question}\n\n"
    content += (
        "Answer the question with reasoning. " if reasoning else "Answer the question. "
    )
    content += "End your response with 'Final Decision: [yes/no/maybe]'."

    messages = [
        {"role": "user", "content": content},
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    outputs = pipe(
        prompt,
        max_new_tokens=512,
        do_sample=False,
        return_full_text=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=(
            tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
        ),
    )

    print(f"Test output of {path}:\n")
    print(outputs[0]["generated_text"])


if __name__ == "__main__":
    # Export SFT version
    merge_and_export(SFT_PATH, SFT_DEPLOYMENT_PATH)
    test_model_deployment(SFT_DEPLOYMENT_PATH, reasoning=False)

    # Export SFT + RLVR version
    merge_and_export(RLVR_PATH, SFT_RLVR_DEPLOYMENT_PATH)
    test_model_deployment(SFT_RLVR_DEPLOYMENT_PATH, reasoning=True)
