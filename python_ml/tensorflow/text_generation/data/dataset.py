from torch.utils.data import Dataset
from dataclasses import dataclass
from datasets import load_dataset

@dataclass
class TextGenerationItem:
    text: str

@dataclass
class DbPediaItem:
    content: str

class DbPediaDataset(Dataset):
    def __init__(self, split: str):
        hf_dataset = load_dataset("dbpedia_14", split=split)
        self.dataset = [DbPediaItem(content=entry["content"]) for entry in hf_dataset]

    @classmethod
    def train(cls):
        return cls("train")

    @classmethod
    def test(cls):
        return cls("test")

    def __getitem__(self, index):
        item = self.dataset[index]
        return TextGenerationItem(text=item.content)

    def __len__(self):
        return len(self.dataset)

