import tensorflow as tf
from .model import TextGenerationModelConfig
from .data.batcher import TextGenerationBatcher
from .data.dataset import DbPediaDataset

def train(dataset_train, dataset_test, config: TextGenerationModelConfig, epochs=5, batch_size=6):
    tokenizer = Gpt2Tokenizer()
    batcher = TextGenerationBatcher(tokenizer, config.max_seq_length)

    model = config.init()

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    for epoch in range(epochs):
        for i in range(0, len(dataset_train), batch_size):
            batch_items = [dataset_train[j] for j in range(i, min(i+batch_size, len(dataset_train)))]
            batch = batcher.batch_train(batch_items)
            with tf.GradientTape() as tape:
                logits, loss = model(batch, training=True)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        print(f"Epoch {epoch+1}/{epochs} done.")
    return model

