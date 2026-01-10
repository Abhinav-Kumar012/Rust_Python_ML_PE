import torch
import torch.nn as nn
from transformers import BertModel


class TextClassificationModel(nn.Module):
    def __init__(self, transformer_config, num_classes, vocab_size, seq_length):
        super().__init__()

        self.bert = BertModel.from_pretrained("bert-base-cased")
        hidden = self.bert.config.hidden_size

        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, batch):
        outputs = self.bert(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        pooled = outputs.pooler_output
        return self.classifier(pooled)

    def infer(self, batch):
        return self.forward(batch)

