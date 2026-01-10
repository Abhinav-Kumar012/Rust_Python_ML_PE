import tensorflow as tf
from model import TextGenerationModelConfig
from data.tokenizer import Gpt2Tokenizer

def generate_text(model, tokenizer, prompt, max_length=50, temperature=1.0):
    input_tokens = tokenizer.encode(prompt, special_tokens=True)
    input_ids = tf.constant([input_tokens], dtype=tf.int32)

    for _ in range(max_length):
        # Compute embeddings
        seq_len = tf.shape(input_ids)[1]
        positions = tf.range(seq_len)[tf.newaxis, :]
        token_emb = model.embedding_token(input_ids)
        pos_emb = model.embedding_pos(positions)
        x = (token_emb + pos_emb) / 2

        for layer in model.transformer_layers:
            x = layer(x)

        logits = model.output_layer(x[:, -1, :])
        probs = tf.nn.softmax(logits / temperature)
        next_token = tf.random.categorical(tf.math.log(probs), 1)
        next_token = tf.cast(next_token, tf.int32)

        input_ids = tf.concat([input_ids, next_token], axis=1)

        if next_token.numpy()[0][0] == tokenizer.end_token():
            break

    return tokenizer.decode(input_ids[0].numpy().tolist())

def main():
    tokenizer = Gpt2Tokenizer()
    config = TextGenerationModelConfig(vocab_size=tokenizer.vocab_size(), pad_token=tokenizer.pad_token(), max_seq_length=128)
    model = config.init()
    model.load_weights("artifacts/tf_model")

    prompt = "Once upon a time"
    text = generate_text(model, tokenizer, prompt)
    print("Generated text:\n", text)

if __name__ == "__main__":
    main()

