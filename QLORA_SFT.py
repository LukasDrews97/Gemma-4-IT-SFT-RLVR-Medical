import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from config import (
    MODEL_ID,
    N_TEST_SAMPLES,
    N_TRAIN_EPOCHS,
    N_TRAIN_SAMPLES,
    SFT_PATH,
    TOKENIZER_ID,
)
from pubmedqa_dataset import get_dataset


def main() -> None:
    train_dataset, test_dataset = get_dataset(
        n_train_samples=N_TRAIN_SAMPLES, n_test_samples=N_TEST_SAMPLES
    )
    print(f"Training on: {len(train_dataset)} samples")
    print(f"Testing on: {len(test_dataset)} expert samples")

    # Print a clear preview of the first training sample
    first_sample = train_dataset[0]["messages"]

    print("\n" + "=" * 50)
    print("--- PROMPT PREVIEW ---")
    print(first_sample[0]["content"])

    print("\n--- TARGET RESPONSE ---")
    print(first_sample[1]["content"])
    print("=" * 50 + "\n")

    # Print the raw list of message dictionaries for debugging
    print("RAW MESSAGE STRUCTURE:")
    for item in first_sample:
        print(item)

    # Define model init arguments
    model_kwargs = dict(
        dtype=torch.bfloat16,
        device_map="auto",
    )

    # BitsAndBytesConfig: Enables 4-bit quantization to reduce model size/memory usage
    model_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.bfloat16,
    )

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=16,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        modules_to_save=["lm_head", "embed_tokens"],
        ensure_weight_tying=True,
    )

    args = SFTConfig(
        output_dir=SFT_PATH,  # directory to save and repository id
        max_length=2048,  # max length for model and packing of the dataset
        num_train_epochs=N_TRAIN_EPOCHS,  # number of training epochs
        completion_only_loss=True,
        per_device_train_batch_size=1,  # batch size per device during training
        gradient_accumulation_steps=4,
        optim="adamw_torch_fused",  # use fused adamw optimizer
        logging_steps=10,  # log every 10 steps
        save_strategy="epoch",  # save checkpoint every epoch
        eval_strategy="epoch",  # evaluate checkpoint every epoch
        learning_rate=5e-5,  # learning rate
        fp16=False,
        bf16=True,  # BF16 strictly enabled
        max_grad_norm=0.3,  # max gradient norm based on QLoRA paper
        lr_scheduler_type="constant",  # use constant learning rate scheduler
        push_to_hub=False,  # push model to hub
        load_best_model_at_end=True,
        dataset_kwargs={
            "add_special_tokens": False,  # Template with special tokens
            "append_concat_token": True,  # Add EOS token as separator token between examples
        },
    )

    # Create Trainer object
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    # Start training
    trainer.train()

    # Save the final model
    trainer.save_model()

    # free the memory again
    del model
    del trainer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
