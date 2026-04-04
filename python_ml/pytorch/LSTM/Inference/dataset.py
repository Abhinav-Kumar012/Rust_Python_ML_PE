import random
import numpy as np
import torch
from torch.utils.data import Dataset

# ==========================================================
# Constants  (mirrors dataset.rs)
# ==========================================================

NUM_SEQUENCES: int = 1000
SEQ_LENGTH: int = 10
NOISE_LEVEL: float = 0.1
RANDOM_SEED: int = 5


# ==========================================================
# SequenceDatasetItem  (mirrors SequenceDatasetItem::new())
# ==========================================================

class SequenceDatasetItem:
    """
    A single sequence where each element is the sum of the previous two
    plus Gaussian noise.  The last element is treated as the prediction target.

    sequence : list[float]  length = SEQ_LENGTH + 1  (all-but-last)
    target   : float        the next value to predict
    """

    def __init__(self, seq_length: int, noise_level: float) -> None:
        # Start with two random numbers in [0, 1)
        seq = [random.random(), random.random()]

        for _ in range(seq_length):
            noise = np.random.normal(0.0, noise_level)
            next_val = seq[-2] + seq[-1] + noise
            seq.append(next_val)

        # All-but-last  →  sequence input
        # Last value    →  target
        self.sequence: list[float] = seq[:-1]
        self.target: float = seq[-1]


# ==========================================================
# SequenceDataset  (mirrors SequenceDataset)
# ==========================================================

class SequenceDataset(Dataset):
    """
    In-memory dataset of SequenceDatasetItems.
    Equivalent to Rust's InMemDataset<SequenceDatasetItem>.
    """

    def __init__(self, num_sequences: int, seq_length: int, noise_level: float) -> None:
        self.items: list[SequenceDatasetItem] = [
            SequenceDatasetItem(seq_length, noise_level)
            for _ in range(num_sequences)
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> SequenceDatasetItem:
        return self.items[index]


# ==========================================================
# Collate / Batcher  (mirrors SequenceBatcher)
# ==========================================================

def collate_fn(items: list[SequenceDatasetItem]):
    """
    Collates a list of SequenceDatasetItems into batched tensors.

    sequences : Tensor[batch_size, seq_length, 1]   (input_size = 1)
    targets   : Tensor[batch_size, 1]
    """
    sequences = torch.tensor(
        [item.sequence for item in items], dtype=torch.float32
    ).unsqueeze(-1)  # (batch, seq_len) → (batch, seq_len, 1)

    targets = torch.tensor(
        [[item.target] for item in items], dtype=torch.float32
    )  # (batch, 1)

    return sequences, targets
