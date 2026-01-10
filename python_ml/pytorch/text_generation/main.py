import torch
from data.dataset import DbPediaDataset
from training import ExperimentConfig, train

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset_train = DbPediaDataset.train()
    dataset_test = DbPediaDataset.test()

    config = ExperimentConfig(
        transformer={
            "d_model": 256,
            "nhead": 8,
            "num_layers": 6,
            "dim_feedforward": 1024,
            "dropout": 0.1,
        },
        optimizer={
            "lr": 0.001
        },
        max_seq_length=128,
        batch_size=6,
        num_epochs=5
    )

    artifact_dir = "artifacts"
    train(device, dataset_train, dataset_test, config, artifact_dir)
    print("Training completed. Artifacts saved in", artifact_dir)

if __name__ == "__main__":
    main()

