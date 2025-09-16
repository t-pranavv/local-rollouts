#!/usr/bin/env python3
"""
download_and_save_hf_models.py

Download a Hugging Face model and tokenizer and save them to the output directory,
creating subfolders based on the model name (e.g. "microsoft/Phi-4-reasoning" → output/microsoft/Phi-4-reasoning).
"""

import os
import sys
import argparse
import logging
from transformers import (
    AutoTokenizer,
    AutoConfig,
    GenerationConfig,
    AutoModelForCausalLM,
)
from transformers.utils import logging as hf_logging


def sanitize_name(name: str) -> str:
    """
    Replace characters that are not filesystem-friendly.
    """
    return name.replace(":", "_")


def resolve_model_class(config) -> object:
    """
    Dynamically choose the correct AutoModel class based on model architecture.
    """
    # if config.is_decoder and not config.is_encoder_decoder:
    #     return AutoModelForCausalLM
    # elif config.is_encoder_decoder:
    #     return AutoModelForSeq2SeqLM
    # else:
    #     return AutoModel
    return AutoModelForCausalLM


def download_and_save(model_name: str, tokenizer_name: str, output_dir: str):
    """
    Download a Hugging Face model, tokenizer, and generation config, then save to output directory.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    hf_logging.set_verbosity_error()

    parts = model_name.split("/")
    sanitized_parts = [sanitize_name(p) for p in parts]
    target_dir = os.path.join(output_dir, *sanitized_parts)
    os.makedirs(target_dir, exist_ok=True)

    logging.info(f"Loading config for '{model_name}'")
    try:
        config = AutoConfig.from_pretrained(model_name)
    except Exception as e:
        logging.error(f"Failed to load config for '{model_name}': {e}")
        sys.exit(1)

    model_class = resolve_model_class(config)
    logging.info(f"Using model class: {model_class.__name__}")

    logging.info(f"Downloading model '{model_name}' to '{target_dir}'")
    try:
        model = model_class.from_pretrained(model_name, config=config)
        model.save_pretrained(target_dir)
        logging.info("Model saved successfully.")
    except Exception as e:
        logging.error(f"Failed to download/save model '{model_name}': {e}")
        sys.exit(1)

    logging.info(f"Downloading tokenizer '{tokenizer_name}' to '{target_dir}'")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        tokenizer.save_pretrained(target_dir)
        logging.info("Tokenizer saved successfully.")
    except Exception as e:
        logging.error(f"Failed to download/save tokenizer '{tokenizer_name}': {e}")
        sys.exit(1)

    logging.info(f"Attempting to download generation config for '{model_name}'")
    try:
        generation_config = GenerationConfig.from_pretrained(model_name)
        generation_config.save_pretrained(target_dir)
        logging.info("Generation config saved successfully.")
    except Exception as e:
        logging.warning(f"Generation config not found or failed to save for '{model_name}': {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="Download and save a Hugging Face model and tokenizer")
    parser.add_argument(
        "--model-name", "-m", required=True, help="Hugging Face model ID (e.g., microsoft/Phi-4-reasoning)"
    )
    parser.add_argument("--tokenizer-name", "-t", default=None, help="Tokenizer ID. Defaults to model name.")
    parser.add_argument("--output-dir", "-o", required=True, help="Directory to save the model and tokenizer")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tokenizer_name = args.tokenizer_name or args.model_name
    download_and_save(args.model_name, tokenizer_name, args.output_dir)
