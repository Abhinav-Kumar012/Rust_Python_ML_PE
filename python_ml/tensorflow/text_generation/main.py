from data.dataset import DbPediaDataset
from training import train
from model import TextGenerationModelConfig

def main():
    dataset_train = DbPediaDataset.train()
    dataset_test = DbPediaDataset.test()

    config = TextGenerationModelConfig(
        vocab_size=50257,  # GPT2 vocab size
        pad_token=50256,   # GPT2 pad token id
        max_seq_length=128,
    )

    model = train(dataset_train, dataset_test, config, epochs=5, batch_size=6)
    model.save_weights("artifacts/tf_model")

if __name__ == "__main__":
    main()

