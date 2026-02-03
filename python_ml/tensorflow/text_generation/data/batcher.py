import tensorflow as tf
from dataclasses import dataclass
from typing import List
from .dataset import TextGenerationItem
from .tokenizer import Tokenizer

@dataclass
class TrainingTextGenerationBatch:
    tokens_inputs: tf.Tensor
    targets: tf.Tensor
    mask_pad: tf.Tensor

class TextGenerationBatcher:
    def __init__(self, tokenizer: Tokenizer, max_seq_length: int):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def batch_train(self, items: List[TextGenerationItem]):
        tokens_list = [self.tokenizer.encode(item.text, special_tokens=True) for item in items]

        batch_size = len(tokens_list)
        seq_length = min(max(len(t) for t in tokens_list), self.max_seq_length)

        tokens_tensor = tf.fill([batch_size, seq_length], self.tokenizer.pad_token())
        mask_pad = tf.zeros([batch_size, seq_length], dtype=tf.bool)

        for i, t in enumerate(tokens_list):
            l = min(len(t), seq_length)
            tokens_tensor = tf.tensor_scatter_nd_update(tokens_tensor,
                                                        indices=[[i, j] for j in range(l)],
                                                        updates=tf.constant(t[:l]))
            mask_pad = tf.tensor_scatter_nd_update(mask_pad,
                                                   indices=[[i, j] for j in range(l)],
                                                   updates=tf.constant([False]*l))

        # Training shift: inputs = tokens[:-1], targets = tokens[1:]
        inputs = tokens_tensor[:, :-1]
        targets = tokens_tensor[:, 1:]
        mask_pad = mask_pad[:, :-1]

        return TrainingTextGenerationBatch(tokens_inputs=inputs, targets=targets, mask_pad=mask_pad)

