class TextClassificationDataset:
    @staticmethod
    def num_classes() -> int:
        raise NotImplementedError

    @staticmethod
    def class_name(index: int) -> str:
        raise NotImplementedError


class AgNewsDataset(TextClassificationDataset):
    CLASSES = ["World", "Sports", "Business", "Sci/Tech"]

    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

    @staticmethod
    def num_classes() -> int:
        return 4

    @staticmethod
    def class_name(index: int) -> str:
        return AgNewsDataset.CLASSES[index]

