import json
import tensorflow as tf

from data.tokenizer import BertCasedTokenizer
from data.batcher import TextClassificationBatcher
from model import TextClassificationModel
from training import ExperimentConfig


def infer(artifact_dir, samples, dataset_cls):
    with open(f"{artifact_dir}/config.json") as f:
        config = ExperimentConfig.from_dict(json.load(f))

    tokenizer = BertCasedTokenizer()
    batcher = TextClassificationBatcher(tokenizer, config.seq_length)

    model = TextClassificationModel(
        config.transformer,
        dataset_cls.num_classes(),
        tokenizer.vocab_size(),
        config.seq_length,
    )

    model.load_weights(f"{artifact_dir}/model")

    batch = batcher.batch(samples)
    logits = model.infer(batch)

    for i, text in enumerate(samples):
        cls = tf.argmax(logits[i]).numpy()
        print(
            f"\nText: {text}\n"
            f"Logits: {logits[i].numpy()}\n"
            f"Prediction: {dataset_cls.class_name(int(cls))}"
        )

