from transformers import BertTokenizer


class Tokenizer:
    def encode(self, text: str):
        raise NotImplementedError

    def vocab_size(self) -> int:
        raise NotImplementedError


class BertCasedTokenizer(Tokenizer):
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    def encode(self, text: str):
        return self.tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=True,
        )

    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

