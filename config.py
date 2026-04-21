"""Training settings."""

from typing import Literal

# Training
MODEL_ID: str = "google/gemma-4-E2B-it"
TOKENIZER_ID: str = "google/gemma-4-E2B-it"
SFT_PATH: str = "./gemma-PubMedQA-SFT"
RLVR_PATH: str = "./gemma-PubMedQA-RLVR"
N_SFT_SAMPLES: int | Literal["all"] = 1024
N_RLVR_SAMPLES: int | Literal["all"] = 256
N_TEST_SAMPLES: int | Literal["all"] = 1000
N_SFT_TRAIN_EPOCHS: int = 3

# Deployment
SFT_DEPLOYMENT_PATH: str = "./finetuned_models/E2B/gemma-4-E2B-it-sft-medical"
SFT_RLVR_DEPLOYMENT_PATH: str = "./finetuned_models/E2B/gemma-4-E2B-it-sft-rlvr-medical"
