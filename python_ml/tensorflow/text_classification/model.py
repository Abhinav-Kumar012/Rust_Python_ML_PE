import tensorflow as tf
from transformers import TFBertModel


class TextClassificationModel(tf.keras.Model):
    def __init__(self, transformer_config, num_classes, vocab_size, seq_length):
        super().__init__()

        self.bert = TFBertModel.from_pretrained("bert-base-cased")
        self.classifier = tf.keras.layers.Dense(num_classes)

    def call(self, inputs, training=False):
        outputs = self.bert(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            training=training,
        )
        pooled = outputs.pooler_output
        return self.classifier(pooled)

    def infer(self, inputs):
        return self.call(inputs, training=False)

