import torch
from training import train, ExperimentConfig
from model import TextClassificationModel
from data.dataset import AgNewsDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data = AgNewsDataset(
    ["World news", "Sports event"],
    [0, 1],
)

test_data = AgNewsDataset(
    ["Business update", "Tech innovation"],
    [2, 3],
)

config = ExperimentConfig(
    transformer={},
    optimizer={"lr": 3e-5},
)

train(
    device=device,
    dataset_train=train_data,
    dataset_test=test_data,
    config=config,
    artifact_dir="artifacts",
    model_cls=TextClassificationModel,
)

