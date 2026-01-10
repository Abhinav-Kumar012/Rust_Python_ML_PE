import json
import math
import os
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from .model import TextGenerationModelConfig
from .data.batcher import TextGenerationBatcher
from .data.tokenizer import Gpt2Tokenizer
from .data.dataset import SamplerDataset

@dataclass
class ExperimentConfig:
    transformer: dict
    optimizer: dict
    max_seq_length: int = 512
    batch_size: int = 6
    num_epochs: int = 50

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)

class NoamLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, model_size, warmup_steps, scale):
        self.model_size = model_size
        self.warmup_steps = warmup_steps
        self.scale = scale
        self.step_num = 0
        super().__init__(optimizer)

    def get_lr(self):
        self.step_num += 1
        factor = min(
            self.step_num ** -0.5,
            self.step_num * self.warmup_steps ** -1.5,
        )
        return [
            self.scale * (self.model_size ** -0.5) * factor
            for _ in self.optimizer.param_groups
        ]

@torch.no_grad()
def validate(model, dataloader):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in dataloader:
        out = model.forward_training(batch)
        total_loss += out["loss"].item() * out["targets"].numel()
        total_tokens += out["targets"].numel()

    perplexity = math.exp(total_loss / total_tokens)
    model.train()
    return perplexity

def train(
    device: torch.device,
    dataset_train,
    dataset_test,
    config: ExperimentConfig,
    artifact_dir: str,
):
    os.makedirs(artifact_dir, exist_ok=True)

    tokenizer = Gpt2Tokenizer()
    batcher = TextGenerationBatcher(tokenizer, config.max_seq_length)

    model = TextGenerationModelConfig(
        transformer=config.transformer,
        vocab_size=tokenizer.vocab_size(),
        pad_token=tokenizer.pad_token(),
        max_seq_length=config.max_seq_length,
    ).init(device)

    dataloader_train = DataLoader(
        SamplerDataset(dataset_train, 10_000),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=batcher.batch_train,
    )

    dataloader_test = DataLoader(
        SamplerDataset(dataset_test, 1_000),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=batcher.batch_train,
    )

    accum = 6
    optim = torch.optim.Adam(model.parameters(), **config.optimizer)

    scheduler = NoamLRScheduler(
        optim,
        model_size=config.transformer["d_model"],
        warmup_steps=6000,
        scale=0.01 / accum,
    )

    model.train()
    global_step = 0
    for epoch in range(config.num_epochs):
        for i, batch in enumerate(dataloader_train):
            out = model.forward_training(batch)
            loss = out["loss"] / accum
            loss.backward()

            if (i + 1) % accum == 0:
                optim.step()
                optim.zero_grad()
                scheduler.step()
                global_step += 1

        validate(model, dataloader_test)

    config.save(os.path.join(artifact_dir, "config.json"))
    torch.save(model.state_dict(), os.path.join(artifact_dir, "model.pt"))

