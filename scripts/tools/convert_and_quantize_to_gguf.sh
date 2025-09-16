#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

CHECKPOINT=""
OUTDIR=""
OUTTYPE="bf16"
CONVERT_SCRIPT="convert_hf_to_gguf.py"
RUN_LLAMA_BENCH=false
QUANTIZATION_TYPE=""

# Default quantization presets
DEFAULT_QUANT_PRESETS=(
    Q2_K Q2_K_S Q3_K Q3_K_S Q3_K_L Q4_0 Q4_1 Q4_K Q4_K_S Q5_0 Q5_1 Q5_K Q5_K_S Q6_K Q8_0
    IQ1_S IQ1_M IQ2_XXS IQ2_XS IQ2_S IQ2_M IQ3_XXS IQ3_XS IQ3_S IQ3_M IQ4_NL IQ4_XS
    TQ1_0 TQ2_0
)

usage() {
    echo "Usage: $0 --checkpoint=<path> --outdir=<path> [--quantization-type=<preset>] [--run-llama-bench]"
    exit 1
}

# Parse and validate command-line arguments
for arg in "$@"; do
    case $arg in
        --checkpoint=*) CHECKPOINT="${arg#*=}" ;;
        --outdir=*) OUTDIR="${arg#*=}" ;;
        --quantization-type=*) QUANTIZATION_TYPE="${arg#*=}" ;;
        --run-llama-bench) RUN_LLAMA_BENCH=true ;;
        *) echo "Unknown argument: $arg"; usage ;;
    esac
done

if [[ -z "$CHECKPOINT" || -z "$OUTDIR" ]]; then
    echo "--checkpoint and --outdir are required."
    usage
fi

# Determine which quantization presets to use
if [[ -n "$QUANTIZATION_TYPE" ]]; then
    VALID=false
    for preset in "${DEFAULT_QUANT_PRESETS[@]}"; do
        if [[ "$preset" == "$QUANTIZATION_TYPE" ]]; then
            VALID=true
            break
        fi
    done

    if ! $VALID; then
        echo "Error: Invalid --quantization-type '$QUANTIZATION_TYPE'"
        echo "Valid options are: ${DEFAULT_QUANT_PRESETS[*]}"
        exit 1
    fi
fi

if [[ -n "$QUANTIZATION_TYPE" ]]; then
    QUANT_PRESETS=("$QUANTIZATION_TYPE")
else
    QUANT_PRESETS=("${DEFAULT_QUANT_PRESETS[@]}")
fi

# Convert HF checkpoint to GGUF format
mkdir -p "$OUTDIR"
BASENAME=$(basename "$CHECKPOINT" | tr ' ' '_' | tr -d '/')
BASE_GGUF="$OUTDIR/${BASENAME}-${OUTTYPE}.gguf"

echo "Preparing $BASE_GGUF..."
if [[ -f "$BASE_GGUF" ]]; then
    echo "Found $BASE_GGUF — skipping conversion..."
else
    echo "Converting $CHECKPOINT to $BASE_GGUF..."
    "$CONVERT_SCRIPT" "$CHECKPOINT" --outfile "$BASE_GGUF" --outtype "$OUTTYPE" || {
        echo "'convert_hf_to_gguf.py' failed for $CHECKPOINT"
        exit 1
    }
fi

# Quantize and optionally run benchmarks
for preset in "${QUANT_PRESETS[@]}"; do
    OUT_Q="$OUTDIR/${BASENAME}-$preset.gguf"
    echo "Quantizing $preset to $OUT_Q..."
    llama-quantize "$BASE_GGUF" "$OUT_Q" "$preset" || {
        echo "'llama-quantize' failed for $preset"
        continue
    }

    if $RUN_LLAMA_BENCH; then
        echo "Running llama-bench for $preset..."
        llama-bench -m "$OUT_Q" || {
            echo "'llama-bench' failed for $preset"
        }
    fi
done

echo "All conversions, quantizations, and benchmarks completed."