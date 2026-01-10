from abc import ABC, abstractmethod
from transformers import GPT2TokenizerFast

class Tokenizer(ABC):
    @abstractmethod
    def encode(self, value: str, special_tokens: bool) -> list[int]:
        pass

    @abstractmethod
    def decode(self, tokens: list[int]) -> str:
        pass

    @abstractmethod
    def vocab_size(self) -> int:
        pass

    @abstractmethod
    def pad_token(self) -> int:
        pass

    @abstractmethod
    def start_token(self) -> int:
        pass

    @abstractmethod
    def end_token(self) -> int:
        pass

    def pad_token_value(self) -> str:
        return self.decode([self.pad_token()])

    def start_token_value(self) -> str:
        return self.decode([self.start_token()])

    def end_token_value(self) -> str:
        return self.decode([self.end_token()])

class Gpt2Tokenizer(Tokenizer):
    def __init__(self):
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.tokenizer.add_special_tokens({
            "pad_token": "[PAD]",
            "bos_token": "[START]",
            "eos_token": "[END]",
        })

    def encode(self, value: str, special_tokens: bool) -> list[int]:
        if special_tokens:
            text = "[START]" + value + "[END]"
        else:
            text = value
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=False)

    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def pad_token(self) -> int:
        return self.tokenizer.convert_tokens_to_ids("[PAD]")

    def start_token(self) -> int:
        return self.tokenizer.convert_tokens_to_ids("[START]")

    def end_token(self) -> int:
        return self.tokenizer.convert_tokens_to_ids("[END]")

