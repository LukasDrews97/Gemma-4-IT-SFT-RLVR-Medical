import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from config import (
    MODEL_ID,
    N_SFT_SAMPLES,
    N_SFT_TRAIN_EPOCHS,
    N_TEST_SAMPLES,
    SFT_PATH,
    TOKENIZER_ID,
    USE_QLORA
)
from pubmedqa_dataset import get_dataset


def main() -> None:
    train_dataset, _, test_dataset = get_dataset(
        n_sft_samples=N_SFT_SAMPLES, n_test_samples=N_TEST_SAMPLES
    )

    if USE_QLORA:
        model_kwargs = dict(
            dtype=torch.bfloat16,
            device_map="auto",
        )

        # 4-bit quantization to reduce model size/memory usage
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage=torch.bfloat16,
        )
    else:
        model_kwargs = {}

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
        **model_kwargs,
    )
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
        output_dir=SFT_PATH,
        max_length=2048,
        num_train_epochs=N_SFT_TRAIN_EPOCHS,
        completion_only_loss=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        optim="adamw_torch_fused",
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        learning_rate=5e-5,
        fp16=False,
        bf16=True,
        max_grad_norm=0.3,  # max gradient norm based on QLoRA paper
        lr_scheduler_type="constant",
        push_to_hub=False,
        load_best_model_at_end=True,
        use_liger_kernel=True,
        dataset_kwargs={
            "add_special_tokens": False,
            "append_concat_token": True,
        },
    )

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
