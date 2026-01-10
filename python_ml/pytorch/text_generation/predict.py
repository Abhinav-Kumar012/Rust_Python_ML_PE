import torch
from model import TextGenerationModelConfig, TextGenerationModel
from data.tokenizer import Gpt2Tokenizer
import argparse

def generate_text(
    model: TextGenerationModel,
    tokenizer: Gpt2Tokenizer,
    prompt: str,
    max_length: int = 50,
    temperature: float = 1.0,
    device: torch.device = torch.device("cpu"),
):
    model.eval()
    input_tokens = tokenizer.encode(prompt, special_tokens=True)
    input_ids = torch.tensor([input_tokens], device=device)

    for _ in range(max_length):
        # Prepare batch
        seq_len = input_ids.shape[1]
        positions = torch.arange(seq_len, device=device).unsqueeze(0)

        token_emb = model.embedding_token(input_ids)
        pos_emb = model.embedding_pos(positions)
        hidden = (token_emb + pos_emb) / 2

        # Create autoregressive mask
        attn_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1,
        )

        output = model.transformer(
            hidden,
            mask=attn_mask
        )

        logits = model.output(output[:, -1, :])
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat([input_ids, next_token], dim=1)

        if next_token.item() == tokenizer.end_token():
            break

    generated_tokens = input_ids[0].tolist()
    return tokenizer.decode(generated_tokens)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True, help="Prompt text")
    parser.add_argument("--model_path", type=str, default="artifacts/model.pt", help="Path to trained model")
    parser.add_argument("--max_length", type=int, default=50, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = Gpt2Tokenizer()
    model = TextGenerationModelConfig(
        transformer={
            "d_model": 256,
            "nhead": 8,
            "num_layers": 6,
            "dim_feedforward": 1024,
            "dropout": 0.1
        },
        vocab_size=tokenizer.vocab_size(),
        pad_token=tokenizer.pad_token(),
        max_seq_length=128
    ).init(device)

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)

    output_text = generate_text(
        model, tokenizer,
        prompt=args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        device=device
    )

    print("\nGenerated text:\n")
    print(output_text)

if __name__ == "__main__":
    main()

