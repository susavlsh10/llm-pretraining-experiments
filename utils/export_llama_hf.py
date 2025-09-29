#!/usr/bin/env python3
"""
Standalone script to export LLaMA models from the custom training format to HuggingFace format.

This script can be used to convert trained models to HuggingFace format and optionally push them to the Hub.

Example usage:

1. Export a local model to HuggingFace format:
   python export_llama_hf.py --input checkpoint.bin --output ./hf_model

2. Export and push to HuggingFace Hub:
   python export_llama_hf.py --input checkpoint.bin --output ./hf_model --push_to_hub --model_name username/my-llama-model

3. Export with custom configuration:
   python export_llama_hf.py --input checkpoint.bin --output ./hf_model --custom_n_layer 12 --custom_n_embd 768

Requirements:
- torch
- transformers
- huggingface_hub (for pushing to Hub)
"""

import argparse
import os
import torch
import json
from pathlib import Path

# Import the necessary classes from the training script
# Note: In practice, you might want to organize these into separate modules
try:
    from train_llama3 import LLaMA, LlamaConfig, export_to_huggingface, test_exported_model
    print("Imported from train_llama3.py successfully")
except ImportError as e:
    print(f"Error importing from train_llama3.py: {e}")
    print("Make sure train_llama3.py is in the same directory or Python path")
    exit(1)

def load_model_from_checkpoint(checkpoint_path, custom_config=None):
    """Load model from a checkpoint file"""
    print(f"Loading model from {checkpoint_path}")
    
    # Load the checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract configuration from checkpoint or use default
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        # Use default config and apply custom overrides
        config = LlamaConfig()
        
    # Apply custom configuration overrides
    if custom_config:
        for key, value in custom_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
                print(f"Applied custom config: {key} = {value}")
    
    # Create model
    model = LLaMA(config)
    
    # Load state dict
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        # Assume the entire checkpoint is the state dict
        model.load_state_dict(checkpoint, strict=False)
    
    print(f"Model loaded with config: {config}")
    return model

def main():
    parser = argparse.ArgumentParser(description="Export LLaMA model to HuggingFace format")
    
    # Input/Output
    parser.add_argument("--input", "-i", type=str, required=True, 
                       help="Path to the trained model checkpoint")
    parser.add_argument("--output", "-o", type=str, required=True,
                       help="Output directory for HuggingFace model")
    
    # HuggingFace Hub options
    parser.add_argument("--push_to_hub", action="store_true",
                       help="Push model to HuggingFace Hub")
    parser.add_argument("--model_name", type=str, default="",
                       help="Model name on HuggingFace Hub (e.g., username/model-name)")
    parser.add_argument("--private_repo", action="store_true",
                       help="Create private repository on HuggingFace Hub")
    
    # Model customization
    parser.add_argument("--custom_n_layer", type=int, default=None,
                       help="Override number of layers")
    parser.add_argument("--custom_n_embd", type=int, default=None,
                       help="Override embedding dimension")
    parser.add_argument("--custom_n_head", type=int, default=None,
                       help="Override number of attention heads")
    parser.add_argument("--custom_n_kv_head", type=int, default=None,
                       help="Override number of key-value heads")
    parser.add_argument("--custom_vocab_size", type=int, default=None,
                       help="Override vocabulary size")
    parser.add_argument("--custom_block_size", type=int, default=None,
                       help="Override maximum sequence length")
    
    # Testing
    parser.add_argument("--test_model", action="store_true", default=True,
                       help="Test the exported model with sample generation")
    parser.add_argument("--test_prompt", type=str, default="The meaning of life is",
                       help="Prompt to use for testing the model")
    
    # Model metadata
    parser.add_argument("--model_description", type=str, default="",
                       help="Description for the model card")
    parser.add_argument("--license", type=str, default="apache-2.0",
                       help="License for the model")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.push_to_hub and not args.model_name:
        print("Error: --model_name is required when --push_to_hub is specified")
        exit(1)
    
    # Gather custom configuration
    custom_config = {}
    for attr in ['custom_n_layer', 'custom_n_embd', 'custom_n_head', 
                 'custom_n_kv_head', 'custom_vocab_size', 'custom_block_size']:
        value = getattr(args, attr)
        if value is not None:
            # Remove 'custom_' prefix
            key = attr.replace('custom_', '')
            custom_config[key] = value
    
    try:
        # Load the model
        model = load_model_from_checkpoint(args.input, custom_config)
        
        # Create a dummy tokenizer if not available
        tokenizer = None
        if hasattr(model, 'tokenizer'):
            tokenizer = model.tokenizer
        
        print("=" * 80)
        print("Exporting to HuggingFace format...")
        
        # Export the model
        hf_model, export_path = export_to_huggingface(
            model=model,
            tokenizer=tokenizer,
            output_dir=args.output,
            model_name=args.model_name if args.model_name else None,
            push_to_hub=args.push_to_hub,
            private_repo=args.private_repo,
            custom_config=custom_config if custom_config else None
        )
        
        # Create a model card
        create_model_card(
            output_dir=args.output,
            model_name=args.model_name,
            description=args.model_description,
            license=args.license,
            config=model.config
        )
        
        print(f"✅ Model exported successfully to: {export_path}")
        
        if args.push_to_hub:
            print(f"✅ Model pushed to HuggingFace Hub: https://huggingface.co/{args.model_name}")
        
        # Test the exported model
        if args.test_model:
            print("=" * 80)
            print("Testing exported model...")
            test_exported_model(export_path, prompt=args.test_prompt)
            
    except Exception as e:
        print(f"❌ Error during export: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

def create_model_card(output_dir, model_name, description, license, config):
    """Create a model card for the exported model"""
    
    model_card_content = f"""---
license: {license}
language:
- en
library_name: transformers
pipeline_tag: text-generation
"""
    
    if model_name:
        model_card_content += f"model-index:\n- name: {model_name}\n"
    
    model_card_content += f"""---

# {model_name if model_name else "Custom LLaMA Model"}

{description if description else "This is a custom LLaMA model trained using the llm-pretraining framework."}

## Model Details

- **Model Type:** LLaMA-based Language Model
- **Architecture:** Transformer with RoPE, GQA, and SwiGLU
- **Parameters:** {get_model_size_estimate(config)}
- **Context Length:** {config.block_size}
- **Vocabulary Size:** {config.vocab_size}

### Architecture Details

- **Layers:** {config.n_layer}
- **Hidden Size:** {config.n_embd}
- **Attention Heads:** {config.n_head}
- **Key-Value Heads:** {config.n_kv_head}
- **FFN Multiplier:** {config.ffn_dim_multiplier}
- **RoPE Theta:** {config.rope_theta}
- **Scaled RoPE:** {config.use_scaled_rope}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained("{model_name if model_name else 'path/to/model'}")
tokenizer = AutoTokenizer.from_pretrained("{model_name if model_name else 'path/to/model'}")

# Generate text
prompt = "The future of artificial intelligence is"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Training

This model was trained using the llm-pretraining framework, which implements:

- RoPE (Rotary Positional Encoding)
- GQA (Grouped Query Attention) 
- SwiGLU activation function
- RMSNorm normalization

## Limitations and Bias

This model may have limitations and biases typical of language models. Users should be aware of potential issues and use the model responsibly.
"""
    
    # Write the model card
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(model_card_content)
    
    print(f"Model card created at {os.path.join(output_dir, 'README.md')}")

def get_model_size_estimate(config):
    """Estimate the number of parameters in the model"""
    # Rough calculation based on typical LLaMA architecture
    vocab_params = config.vocab_size * config.n_embd
    embed_params = config.block_size * config.n_embd  # position embeddings (if used)
    
    # Per layer: attention + MLP + norms
    hidden_size = config.n_embd
    intermediate_size = int(hidden_size * config.ffn_dim_multiplier * 4 * 2 / 3)
    
    attention_params = hidden_size * (config.n_head + 2 * config.n_kv_head) * (hidden_size // config.n_head)
    attention_params += hidden_size * hidden_size  # output projection
    
    mlp_params = hidden_size * intermediate_size * 3  # gate, up, down projections
    norm_params = hidden_size * 2  # input and post attention layer norms
    
    layer_params = attention_params + mlp_params + norm_params
    total_params = vocab_params + config.n_layer * layer_params + hidden_size  # final norm
    
    if total_params < 1e6:
        return f"{total_params/1e3:.1f}K"
    elif total_params < 1e9:
        return f"{total_params/1e6:.1f}M"
    else:
        return f"{total_params/1e9:.1f}B"

if __name__ == "__main__":
    main()
