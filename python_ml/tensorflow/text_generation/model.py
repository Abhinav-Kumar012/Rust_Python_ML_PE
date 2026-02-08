import tensorflow as tf
from tensorflow.keras import layers
from dataclasses import dataclass
from .data.batcher import TrainingTextGenerationBatch

@dataclass
class TextGenerationModelConfig:
    vocab_size: int
    pad_token: int
    max_seq_length: int
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.1

    def init(self):
        return TextGenerationModel(self)

class TextGenerationModel(tf.keras.Model):
    def __init__(self, config: TextGenerationModelConfig):
        super().__init__()
        self.config = config
        self.embedding_token = layers.Embedding(config.vocab_size, config.d_model)
        self.embedding_pos = layers.Embedding(config.max_seq_length, config.d_model)

        self.transformer_layers = [
            layers.TransformerEncoder(
                num_heads=config.nhead,
                intermediate_dim=config.dim_feedforward,
                dropout=config.dropout,
                activation="relu"
            ) for _ in range(config.num_layers)
        ]

        self.output_layer = layers.Dense(config.vocab_size)

    def call(self, batch: TrainingTextGenerationBatch, training=True):
        x = batch.tokens_inputs
        seq_len = tf.shape(x)[1]

        positions = tf.range(seq_len)[tf.newaxis, :]
        token_emb = self.embedding_token(x)
        pos_emb = self.embedding_pos(positions)
        x_emb = (token_emb + pos_emb) / 2

        # Autoregressive mask
        attn_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        for layer in self.transformer_layers:
            x_emb = layer(x_emb, mask=attn_mask, training=training)

        logits = self.output_layer(x_emb)
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            batch.targets, logits, from_logits=True
        )
        mask = tf.cast(~batch.mask_pad, tf.float32)
        loss = tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)

        return logits, loss

