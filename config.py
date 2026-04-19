"""Training settings."""

from typing import Literal

MODEL_ID: str = "google/gemma-4-E2B-it"
TOKENIZER_ID: str = "google/gemma-4-E2B-it"
SFT_PATH: str = "./gemma-PubMedQA-SFT"
RLVR_PATH: str = "./gemma-PubMedQA-RLVR"
N_TRAIN_SAMPLES: int | Literal["all"] = 100
N_TEST_SAMPLES: int | Literal["all"] = 100
N_TRAIN_EPOCHS: int = 3
