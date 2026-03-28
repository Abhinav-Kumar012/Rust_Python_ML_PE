from dataclasses import dataclass

from dataset import RANDOM_SEED


# ==========================================================
# TrainingConfig  (mirrors training.rs TrainingConfig)
# ==========================================================

@dataclass
class TrainingConfig:
    num_epochs: int   = 30
    batch_size: int   = 32
    num_workers: int  = 2
    lr: float         = 1e-3
    seed: int         = RANDOM_SEED

    # Model hyper-parameters (mirrors LstmNetworkConfig defaults)
    input_size: int   = 1
    hidden_size: int  = 32
    num_layers: int   = 2
    output_size: int  = 1
    dropout: float    = 0.1
    bidirectional: bool = True
