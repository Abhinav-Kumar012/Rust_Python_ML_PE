import tensorflow as tf


class TextClassificationBatcher:
    def __init__(self, tokenizer, seq_length):
        self.tokenizer = tokenizer
        self.seq_length = seq_length

    def batch(self, samples):
        input_ids = []
        attention_masks = []

        for text in samples:
            ids = self.tokenizer.encode(text)
            ids = ids[: self.seq_length]
            mask = [1] * len(ids)

            padding = self.seq_length - len(ids)
            if padding > 0:
                ids += [0] * padding
                mask += [0] * padding

            input_ids.append(ids)
            attention_masks.append(mask)

        return {
            "input_ids": tf.convert_to_tensor(input_ids, dtype=tf.int32),
            "attention_mask": tf.convert_to_tensor(attention_masks, dtype=tf.int32),
        }

    def batch_with_labels(self, batch):
        texts, labels = zip(*batch)
        inputs = self.batch(texts)
        labels = tf.convert_to_tensor(labels, dtype=tf.int32)
        return inputs, labels

