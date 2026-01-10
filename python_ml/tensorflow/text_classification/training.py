import json
import tensorflow as tf
from tqdm import tqdm

from data.tokenizer import BertCasedTokenizer
from data.batcher import TextClassificationBatcher


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
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=config.optimizer["lr"])
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch}")
        for i in tqdm(range(0, len(dataset_train), config.batch_size)):
            batch = [
                dataset_train[j]
                for j in range(i, min(i + config.batch_size, len(dataset_train)))
            ]
            inputs, labels = batcher.batch_with_labels(batch)

            with tf.GradientTape() as tape:
                logits = model(inputs, training=True)
                loss = loss_fn(labels, logits)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

    config.save(f"{artifact_dir}/config.json")
    model.save_weights(f"{artifact_dir}/model")

