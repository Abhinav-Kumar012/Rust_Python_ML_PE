import json
import torch

from data.tokenizer import BertCasedTokenizer
from data.batcher import TextClassificationBatcher
from model import TextClassificationModel
from training import ExperimentConfig


def infer(device, artifact_dir, samples, dataset_cls):
    with open(f"{artifact_dir}/config.json") as f:
        config = ExperimentConfig.from_dict(json.load(f))

    tokenizer = BertCasedTokenizer()
    batcher = TextClassificationBatcher(tokenizer, config.seq_length)

    model = TextClassificationModel(
        config.transformer,
        dataset_cls.num_classes(),
        tokenizer.vocab_size(),
        config.seq_length,
    ).to(device)

    model.load_state_dict(
        torch.load(f"{artifact_dir}/model.pt", map_location=device)
    )
    model.eval()

    batch = batcher.batch(samples, device)

    with torch.no_grad():
        logits = model.infer(batch)

    for i, text in enumerate(samples):
        cls = logits[i].argmax().item()
        print(
            f"\nText: {text}\n"
            f"Logits: {logits[i].cpu().numpy()}\n"
            f"Prediction: {dataset_cls.class_name(cls)}"
        )

