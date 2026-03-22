import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class TextGenerationModelConfig:
    vocab_size: int
    pad_token: int
    max_seq_length: int
    d_model: int = 384
    nhead: int = 12
    num_layers: int = 6
    dim_feedforward: int = 1536
    norm_first: bool = True
    dropout: float = 0.1  # Burn default is usually 0.1 depending on version, generic sane default

class TextGenerationModel(nn.Module):
    def __init__(self, config: TextGenerationModelConfig):
        super().__init__()
        self.config = config
        
        self.embedding_token = nn.Embedding(config.vocab_size, config.d_model)
        self.embedding_pos = nn.Embedding(config.max_seq_length, config.d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            norm_first=config.norm_first
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        
        self.output = nn.Linear(config.d_model, config.vocab_size)
        
    def forward(self, inputs, mask_pad=None):
        """
        inputs: [batch_size, seq_length] (LongTensor)
        mask_pad: [batch_size, seq_length] (BoolTensor) - True where PADDING is.
        """
        batch_size, seq_length = inputs.shape
        device = inputs.device
        
        # Position Embeddings
        # Rust: arange(0..seq_length).reshape([1, seq_length]).repeat_dim(0, batch_size)
        positions = torch.arange(seq_length, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        # Rust: (embedding_positions + embedding_tokens) / 2
        # Note: Burn adds them and divides by 2?
        # Line 74 in model.rs: let embedding = (embedding_positions + embedding_tokens) / 2;
        # Yes, explicitly divides by 2.
        
        emb_token = self.embedding_token(inputs)
        emb_pos = self.embedding_pos(positions)
        embedding = (emb_token + emb_pos) / 2
        
        # Masks
        # Causal Mask (autoregressive)
        # Rust: generate_autoregressive_mask
        # PyTorch: generate_square_subsequent_mask returns float mask (-inf for future)
        mask_attn = nn.Transformer.generate_square_subsequent_mask(seq_length, device=device)
        
        # Padding Mask
        # PyTorch TransformerEncoder takes src_key_padding_mask
        # If mask_pad is provided (True for PAD), we can pass it directly as src_key_padding_mask
        
        encoded = self.transformer(
            embedding, 
            mask=mask_attn, 
            src_key_padding_mask=mask_pad,
            is_causal=True
        )
        
        output = self.output(encoded)
        return output

    def loss(self, logits, targets, pad_token=None):
        """
        logits: [batch_size, seq_length, vocab_size]
        targets: [batch_size, seq_length]
        """
        # Flatten
        logits_flat = logits.view(-1, self.config.vocab_size)
        targets_flat = targets.view(-1)
        
        ignore_index = pad_token if pad_token is not None else self.config.pad_token
        
        return nn.functional.cross_entropy(logits_flat, targets_flat, ignore_index=ignore_index)
