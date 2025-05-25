#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LoRA Model Merging and Testing Script

This script merges LoRA adapters with base Mistral-7B model and performs domain-specific testing.
Supports medical, legal, and combined domain distillation.

License: MIT License
Copyright (c) 2023 Your Organization
"""

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import os

DOMAIN_TEST_CASES = {
    'med_distill': [
        "Explain the mechanism of NSAIDs in pain relief.",
        "Describe the pathophysiology of type 2 diabetes.",
        "What are the first-line treatments for hypertension?"
    ],
    'legal_distill': [
        "Explain the doctrine of precedent in common law systems.",
        "What are the key elements of a valid contract?",
        "Describe the difference between civil and criminal law."
    ],
    'combined_distill': [
        "Explain how medical malpractice cases are evaluated in court.",
        "Describe the legal requirements for pharmaceutical drug approval.",
        "What are the liability considerations for telemedicine providers?"
    ]
}


def merge_and_save_model(base_model_path: str,
                         lora_path: str,
                         save_path: str,
                         device: str = "cuda:0") -> tuple:
    """Merge LoRA adapters with base model and save the result.

    Args:
        base_model_path: Path to base model
        lora_path: Path to LoRA adapters
        save_path: Output directory for merged model
        device: Target device for model loading

    Returns:
        Tuple of (merged_model, tokenizer)
    """
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map=device
    )

    # Merge with LoRA adapters
    merged_model = PeftModel.from_pretrained(
        base_model,
        lora_path
    ).merge_and_unload()

    # Save merged model
    os.makedirs(save_path, exist_ok=True)
    merged_model.save_pretrained(save_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(save_path)

    return merged_model, tokenizer


def verify_model_load(model_path: str,
                      device: str = "cuda:3") -> tuple:
    """Verify model can be loaded correctly.

    Args:
        model_path: Path to saved model
        device: Target device for verification

    Returns:
        Tuple of (verified_model, verified_tokenizer)
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def create_generation_pipeline(model, tokenizer, **kwargs):
    """Create text generation pipeline with default settings."""
    return pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        do_sample=True,
        temperature=0.7,
        max_new_tokens=512,
        **kwargs
    )


def test_domain_performance(pipeline, domain: str):
    """Test model performance on domain-specific questions."""
    print(f"\n{'=' * 30} {domain.upper()} DOMAIN TESTING {'=' * 30}")
    for i, question in enumerate(DOMAIN_TEST_CASES[domain], 1):
        print(f"\nTest Case {i}: {question}")
        response = pipeline(question)[0]
        print(f"\nResponse:\n{response['generated_text']}")


def main():
    # Configuration
    your_path = "..."
    BASE_MODEL_PATH = f"{your_path}/DivScore/models/Mistral-7B-v0.2/"
    EPOCHS = [8]  # Optimized epoch number Can be extended to multiple epochs

    for distill_type in ['combined_distill', 'legal_distill', 'med_distill']:
        for epoch in EPOCHS:
            print(f"\nProcessing: Epoch {epoch}, Distillation Type: {distill_type}")

            # Path configuration
            lora_path = f"{your_path}/DivScore/experiments/domain_distillation/{distill_type}/mistral-7B-epoch{epoch}"
            save_path = f"{your_path}/DivScore/experiments/domain_distillation/{distill_type}/models/Mistral-7B-v0.2-{distill_type}-epoch{epoch}"

            # Model processing pipeline
            try:
                # 1. Merge and save model
                merged_model, tokenizer = merge_and_save_model(
                    BASE_MODEL_PATH,
                    lora_path,
                    save_path
                )

                # 2. Verify model loading
                test_model, test_tokenizer = verify_model_load(save_path)

                # 3. Create generation pipeline
                generator = create_generation_pipeline(test_model, test_tokenizer)

                # 4. Domain-specific testing
                test_domain_performance(generator, distill_type)

            except Exception as e:
                print(f"Error processing {distill_type} epoch {epoch}: {str(e)}")
                continue


if __name__ == "__main__":
    main()
