import torch
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import TextGenerationModel, TextGenerationModelConfig
from data import TextGenerationData

def verify():
    print("Verifying Model Architecture...")
    
    # Initialize Config
    # Default Rust config: d_model=384, etc.
    config = TextGenerationModelConfig(
        vocab_size=50260, # Approximation
        pad_token=50259,
        max_seq_length=512
    )
    
    model = TextGenerationModel(config)
    
    print(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params}")
    
    # Check specific layer properties
    print(f"d_model: {model.config.d_model}")
    print(f"nhead: {model.config.nhead}")
    print(f"num_layers: {model.config.num_layers}")
    print(f"norm_first: {model.config.norm_first}")
    
    # Check Transformer structure
    print(f"Transformer Norm First: {model.transformer.layers[0].norm1.eps if hasattr(model.transformer.layers[0], 'norm1') else 'N/A'}")
    # PyTorch TransformerEncoderLayer structure:
    # if norm_first: norm1 -> attn -> norm2 -> mlp
    
    print("Verification Script Finished.")

if __name__ == "__main__":
    verify()
