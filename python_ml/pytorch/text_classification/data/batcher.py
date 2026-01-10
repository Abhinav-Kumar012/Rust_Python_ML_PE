import torch


class TextClassificationBatcher:
    def __init__(self, tokenizer, seq_length):
        self.tokenizer = tokenizer
        self.seq_length = seq_length

    def batch(self, samples, device):
        token_ids = []
        attention_masks = []

        for text in samples:
            ids = self.tokenizer.encode(text)
            ids = ids[: self.seq_length]
            mask = [1] * len(ids)

            padding = self.seq_length - len(ids)
            if padding > 0:
                ids += [0] * padding
                mask += [0] * padding

            token_ids.append(ids)
            attention_masks.append(mask)

        return {
            "input_ids": torch.tensor(token_ids, device=device),
            "attention_mask": torch.tensor(attention_masks, device=device),
        }

    def batch_with_labels(self, batch, device):
        texts, labels = zip(*batch)
        inputs = self.batch(texts, device)
        labels = torch.tensor(labels, device=device)
        return inputs, labels

