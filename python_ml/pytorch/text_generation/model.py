import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from .data.batcher import TrainingTextGenerationBatch

@dataclass
class TextGenerationModelConfig:
    transformer: dict
    vocab_size: int
    pad_token: int
    max_seq_length: int

    def init(self, device: torch.device):
        return TextGenerationModel(
            transformer_config=self.transformer,
            vocab_size=self.vocab_size,
            pad_token=self.pad_token,
            max_seq_length=self.max_seq_length,
        ).to(device)

class TextGenerationModel(nn.Module):
    def __init__(
        self,
        transformer_config: dict,
        vocab_size: int,
        pad_token: int,
        max_seq_length: int,
    ):
        super().__init__()

        d_model = transformer_config["d_model"]
        nhead = transformer_config["nhead"]
        num_layers = transformer_config["num_layers"]
        dim_feedforward = transformer_config.get("dim_feedforward", 2048)
        dropout = transformer_config.get("dropout", 0.0)

        self.embedding_token = nn.Embedding(vocab_size, d_model)
        self.embedding_pos = nn.Embedding(max_seq_length, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.output = nn.Linear(d_model, vocab_size)

        self.vocab_size = vocab_size
        self.pad_token = pad_token
        self.max_seq_length = max_seq_length

    def _generate_autoregressive_mask(self, seq_len: int, device: torch.device):
        return torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1,
        )

    def forward_training(
        self,
        item: TrainingTextGenerationBatch,
    ):
        tokens_inputs = item.tokens_inputs
        targets = item.targets
        mask_pad = item.mask_pad

        batch_size, seq_length = tokens_inputs.shape
        device = tokens_inputs.device

        index_positions = (
            torch.arange(seq_length, device=device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )

        embedding_pos = self.embedding_pos(index_positions)
        embedding_tok = self.embedding_token(tokens_inputs)

        embedding = (embedding_pos + embedding_tok) / 2

        attn_mask = self._generate_autoregressive_mask(seq_length, device)
        key_padding_mask = mask_pad.bool()

        encoded = self.transformer(
            embedding,
            mask=attn_mask,
            src_key_padding_mask=key_padding_mask,
        )

        output = self.output(encoded)

        output_flat = output.reshape(batch_size * seq_length, self.vocab_size)
        targets_flat = targets.reshape(batch_size * seq_length)

        loss = F.cross_entropy(
            output_flat,
            targets_flat,
            ignore_index=self.pad_token,
        )

        return {
            "loss": loss,
            "output": output_flat,
            "targets": targets_flat,
        }

