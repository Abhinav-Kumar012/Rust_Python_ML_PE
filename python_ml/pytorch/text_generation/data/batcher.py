import torch
from dataclasses import dataclass
from typing import List
from .dataset import TextGenerationItem
from .tokenizer import Tokenizer

@dataclass
class TextGenerationBatch:
    tokens: torch.Tensor
    mask_pad: torch.Tensor

@dataclass
class TrainingTextGenerationBatch:
    tokens_inputs: torch.Tensor
    targets: torch.Tensor
    mask_pad: torch.Tensor

class TextGenerationBatcher:
    def __init__(self, tokenizer: Tokenizer, max_seq_length: int):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def batch(self, items: List[TextGenerationItem], device: torch.device):
        tokens_list = [self.tokenizer.encode(item.text, add_special_tokens=True) for item in items]

        batch_size = len(tokens_list)
        seq_length = min(max(len(t) for t in tokens_list), self.max_seq_length)
        tokens_tensor = torch.full((batch_size, seq_length), self.tokenizer.pad_token(), dtype=torch.long, device=device)
        mask_pad = torch.zeros((batch_size, seq_length), dtype=torch.bool, device=device)

        for i, t in enumerate(tokens_list):
            l = min(len(t), seq_length)
            tokens_tensor[i, :l] = torch.tensor(t[:l], device=device)
            mask_pad[i, l:] = True

        return TextGenerationBatch(tokens=tokens_tensor, mask_pad=mask_pad)

    def batch_train(self, items: List[TextGenerationItem], device: torch.device):
        item_batch = self.batch(items, device)
        inputs = item_batch.tokens[:, :-1]
        targets = item_batch.tokens[:, 1:]
        mask_pad = item_batch.mask_pad[:, :-1]
        return TrainingTextGenerationBatch(tokens_inputs=inputs, targets=targets, mask_pad=mask_pad)

