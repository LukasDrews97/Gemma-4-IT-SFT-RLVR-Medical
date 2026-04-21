#!/bin/bash

if [ ! -d "llama.cpp" ]; then
    echo "--- llama.cpp not found. Cloning and building... ---"
    git clone https://github.com/ggml-org/llama.cpp.git
    cd llama.cpp
    uv pip install -r requirements.txt
    cmake -B build
    cmake --build build --config Release -t llama-quantize -j
else
    cd llama.cpp
    uv pip install -r requirements.txt
fi

INPUT_DIR="../finetuned_models/E2B"
OUTPUT_DIR="../finetuned_models/E2B/GGUF"

mkdir -p "$INPUT_DIR"
mkdir -p "$OUTPUT_DIR"

MODELS=(
    "gemma-4-E2B-it-sft-medical"
    "gemma-4-E2B-it-sft-rlvr-medical"
)

QUANTS=(
    "Q8_0" 
    "Q6_K" 
    "Q5_K_M" 
    "Q4_K_M"
)

for MODEL in "${MODELS[@]}"; do
    echo "--- Converting $MODEL to GGUF F16 ---"

    uv run python convert_hf_to_gguf.py \
            "$INPUT_DIR/$MODEL" \
            --outfile "$OUTPUT_DIR/$MODEL-F16.gguf" \
            --outtype f16

    for Q in "${QUANTS[@]}"; do
            echo "--- Quantizing to $Q ---"
            ./build/bin/llama-quantize "$OUTPUT_DIR/$MODEL-F16.gguf" "$OUTPUT_DIR/$MODEL-$Q.gguf" "$Q"
    done

    echo "Done with $MODEL"
    echo ""
done
