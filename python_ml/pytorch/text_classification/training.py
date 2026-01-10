import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

from data.batcher import TextClassificationBatcher
from data.tokenizer import BertCasedTokenizer


class ExperimentConfig:
    def __init__(
        self,
        transformer,
        optimizer,
        seq_length=256,
        batch_size=8,
        num_epochs=5,
    ):
        self.transformer = transformer
        self.optimizer = optimizer
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.__dict__, f)

    @staticmethod
    def from_dict(d):
        return ExperimentConfig(**d)


def train(
    device,
    dataset_train,
    dataset_test,
    config: ExperimentConfig,
    artifact_dir: str,
    model_cls,
):
    tokenizer = BertCasedTokenizer()
    batcher = TextClassificationBatcher(tokenizer, config.seq_length)

    model = model_cls(
        config.transformer,
        dataset_train.num_classes(),
        tokenizer.vocab_size(),
        config.seq_length,
    ).to(device)

    train_loader = DataLoader(
        dataset_train,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda b: batcher.batch_with_labels(b, device),
    )

    test_loader = DataLoader(
        dataset_test,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=lambda b: batcher.batch_with_labels(b, device),
    )

    optimizer = Adam(model.parameters(), lr=config.optimizer["lr"])
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(config.num_epochs):
        model.train()
        for batch, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
            optimizer.zero_grad()
            logits = model(batch)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

    config.save(f"{artifact_dir}/config.json")
    torch.save(model.state_dict(), f"{artifact_dir}/model.pt")

