#!/usr/bin/env python3
"""
Convert trained GPT model to HuggingFace format.

This script converts your trained GPT model weights to HuggingFace format,
enabling easy deployment and inference with the transformers library.
"""

import os
import json
import torch
import argparse
from pathlib import Path
from typing import Dict, Any
from dataclasses import asdict

# Import your model classes
from gpt import GPT, GPTConfig

# HuggingFace imports
try:
    from transformers import (
        GPT2LMHeadModel, 
        GPT2Config, 
        GPT2Tokenizer,
        AutoTokenizer,
        AutoModelForCausalLM
    )
except ImportError:
    print("Please install transformers: pip install transformers")
    exit(1)


def convert_state_dict_to_hf(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Convert your GPT state dict to HuggingFace GPT2 format.
    
    Your model uses the same parameter names as HuggingFace GPT2, so this is mostly
    a direct mapping with some key adjustments.
    """
    hf_state_dict = {}
    
    for key, tensor in state_dict.items():
        # Direct mappings - your model already uses HF-compatible names
        if key.startswith('transformer.'):
            hf_key = key
        elif key == 'lm_head.weight':
            # In HuggingFace, lm_head and wte often share weights
            hf_key = key
        else:
            hf_key = key
            
        hf_state_dict[hf_key] = tensor
        
    # HuggingFace GPT2 expects some specific weight tying
    # Your model already does weight tying, so we maintain that
    if 'lm_head.weight' in hf_state_dict and 'transformer.wte.weight' in hf_state_dict:
        # Verify they're the same (should be due to weight tying)
        if not torch.equal(hf_state_dict['lm_head.weight'], hf_state_dict['transformer.wte.weight']):
            print("Warning: lm_head and wte weights are not tied. Using lm_head weights.")
    
    return hf_state_dict


def create_hf_config(gpt_config: GPTConfig) -> GPT2Config:
    """
    Convert your GPTConfig to HuggingFace GPT2Config.
    """
    return GPT2Config(
        vocab_size=gpt_config.vocab_size,
        n_positions=gpt_config.block_size,
        n_embd=gpt_config.n_embd,
        n_layer=gpt_config.n_layer,
        n_head=gpt_config.n_head,
        activation_function="gelu_new",  # Your model uses NewGELU
        resid_pdrop=0.0,  # No dropout in your model
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        summary_type="cls_index",
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.0,
        scale_attn_weights=True,
        use_cache=True,
        scale_attn_by_inverse_layer_idx=False,
        reorder_and_upcast_attn=False,
        # Architecture specific
        architectures=["GPT2LMHeadModel"],
        model_type="gpt2",
        torch_dtype="float32",  # or "bfloat16" based on your training
    )


def convert_to_hf_format(
    model_path: str,
    output_dir: str,
    model_name: str = "custom-gpt2",
    push_to_hub: bool = False,
    hub_model_id: str = None
):
    """
    Convert your trained GPT model to HuggingFace format.
    
    Args:
        model_path: Path to your saved model checkpoint
        output_dir: Directory to save the HuggingFace model
        model_name: Name for the model
        push_to_hub: Whether to push to HuggingFace Hub
        hub_model_id: Model ID on HuggingFace Hub (username/model-name)
    """
    
    # Load your trained model
    print(f"Loading model from {model_path}")
    
    if model_path.endswith('.pt') or model_path.endswith('.pth'):
        # PyTorch checkpoint format
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
            config = checkpoint.get('config', None)
        else:
            state_dict = checkpoint
            config = None
    else:
        # Assume it's a saved model directory or state dict
        state_dict = torch.load(model_path, map_location='cpu')
        config = None
    
    # If config is not in checkpoint, create a default one
    if config is None:
        print("Config not found in checkpoint, using default GPT2-small config")
        config = GPTConfig()  # Uses default values
    
    # Convert state dict to HuggingFace format
    print("Converting state dict to HuggingFace format...")
    hf_state_dict = convert_state_dict_to_hf(state_dict)
    
    # Create HuggingFace config
    print("Creating HuggingFace config...")
    hf_config = create_hf_config(config)
    
    # Create HuggingFace model
    print("Creating HuggingFace model...")
    hf_model = GPT2LMHeadModel(hf_config)
    
    # Load the converted state dict
    print("Loading converted weights...")
    missing_keys, unexpected_keys = hf_model.load_state_dict(hf_state_dict, strict=False)
    
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")
    
    # Prepare output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save the model
    print(f"Saving HuggingFace model to {output_path}")
    hf_model.save_pretrained(output_path)
    hf_config.save_pretrained(output_path)
    
    # Create tokenizer (using GPT-2 tokenizer as base)
    print("Setting up tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.save_pretrained(output_path)
    
    # Create model card
    model_card_content = f"""---
language: en
license: mit
tags:
- gpt2
- text-generation
- pytorch
model_name: {model_name}
---

# {model_name}

This is a GPT-2 style language model converted from a custom training setup.

## Model Details

- **Model Type:** GPT-2
- **Parameters:** {sum(p.numel() for p in hf_model.parameters()):,}
- **Architecture:** 
  - Layers: {hf_config.n_layer}
  - Hidden Size: {hf_config.n_embd}
  - Attention Heads: {hf_config.n_head}
  - Vocab Size: {hf_config.vocab_size}
  - Context Length: {hf_config.n_positions}

## Usage

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained('{hub_model_id or model_name}')
tokenizer = GPT2Tokenizer.from_pretrained('{hub_model_id or model_name}')

# Generate text
input_text = "Hello, world!"
inputs = tokenizer(input_text, return_tensors='pt')
outputs = model.generate(**inputs, max_length=100, do_sample=True, temperature=0.8)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

## Training

This model was trained using a custom GPT implementation.
"""
    
    with open(output_path / "README.md", "w") as f:
        f.write(model_card_content)
    
    # Create generation config
    generation_config = {
        "do_sample": True,
        "max_length": 1024,
        "temperature": 0.8,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "bos_token_id": tokenizer.bos_token_id,
    }
    
    with open(output_path / "generation_config.json", "w") as f:
        json.dump(generation_config, f, indent=2)
    
    print(f"✅ Model converted successfully!")
    print(f"📁 Output directory: {output_path}")
    print(f"🔢 Model parameters: {sum(p.numel() for p in hf_model.parameters()):,}")
    
    # Test the converted model
    print("\n🧪 Testing the converted model...")
    test_conversion(output_path)
    
    # Push to Hub if requested
    if push_to_hub and hub_model_id:
        print(f"\n🚀 Pushing to HuggingFace Hub: {hub_model_id}")
        try:
            hf_model.push_to_hub(hub_model_id)
            tokenizer.push_to_hub(hub_model_id)
            print(f"✅ Successfully pushed to https://huggingface.co/{hub_model_id}")
        except Exception as e:
            print(f"❌ Failed to push to Hub: {e}")
            print("Make sure you're logged in with `huggingface-cli login`")
    
    return output_path


def test_conversion(model_path: str):
    """Test that the converted model works correctly."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Load the converted model
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Test generation
        test_prompt = "The quick brown fox"
        inputs = tokenizer(test_prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=50,
                do_sample=True,
                temperature=0.8,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ Test successful!")
        print(f"📝 Input: {test_prompt}")
        print(f"📝 Output: {generated_text}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")


def convert_from_checkpoint(checkpoint_path: str, output_dir: str, **kwargs):
    """
    Convert from a training checkpoint that contains model, config, and optimizer state.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract just the model state dict and config
    model_state = checkpoint.get('model', checkpoint)
    config = checkpoint.get('config', GPTConfig())
    
    # Create a temporary dict to pass to main conversion function
    temp_checkpoint = {
        'model': model_state,
        'config': config
    }
    
    # Save temporarily and convert
    temp_path = Path(output_dir) / "temp_model.pt"
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(temp_checkpoint, temp_path)
    
    try:
        result = convert_to_hf_format(str(temp_path), output_dir, **kwargs)
        temp_path.unlink()  # Clean up temp file
        return result
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise e


def main():
    parser = argparse.ArgumentParser(description="Convert GPT model to HuggingFace format")
    parser.add_argument("--model_path", help="Path to your trained model checkpoint")
    parser.add_argument("--output_dir", help="Directory to save HuggingFace model")
    parser.add_argument("--model-name", default="custom-gpt2", help="Name for the model")
    parser.add_argument("--push-to-hub", action="store_true", help="Push to HuggingFace Hub")
    parser.add_argument("--hub-model-id", help="Model ID on HuggingFace Hub (username/model-name)")
    parser.add_argument("--from-checkpoint", action="store_true", 
                       help="Convert from training checkpoint (contains optimizer state)")
    
    args = parser.parse_args()
    
    if args.push_to_hub and not args.hub_model_id:
        parser.error("--hub-model-id is required when --push-to-hub is specified")
    
    print("🔄 Starting GPT to HuggingFace conversion...")
    
    if args.from_checkpoint:
        convert_from_checkpoint(
            args.model_path,
            args.output_dir,
            model_name=args.model_name,
            push_to_hub=args.push_to_hub,
            hub_model_id=args.hub_model_id
        )
    else:
        convert_to_hf_format(
            args.model_path,
            args.output_dir,
            model_name=args.model_name,
            push_to_hub=args.push_to_hub,
            hub_model_id=args.hub_model_id
        )


if __name__ == "__main__":
    main()
