import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import GPT2Tokenizer

class TextGenerationData:
    def __init__(self, model_name="gpt2", max_seq_length=512):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        # Add special tokens exactly as Rust implementation does
        # Rust: [START], [END], [PAD]
        # GPT2 default special tokens are different, so we add custom ones.
        self.tokenizer.add_special_tokens({
            "bos_token": "[START]",
            "eos_token": "[END]",
            "pad_token": "[PAD]",
            "additional_special_tokens": ["[START]", "[END]", "[PAD]"] # ensuring they are treated as special
        })
        self.max_seq_length = max_seq_length

    def load_dbpedia(self, split="train"):
        # Rust uses "dbpedia_14".
        # Burn HuggingfaceDatasetLoader: new("dbpedia_14").dataset(split)
        # We use `datasets` library.
        dataset = load_dataset("dbpedia_14", split=split)
        return dataset

    def get_vocab_size(self):
        return len(self.tokenizer)

    def get_pad_token_id(self):
        return self.tokenizer.pad_token_id

class CollateFn:
    def __init__(self, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __call__(self, batch):
        # batch is list of items from dataset. DbPedia has 'content' field.
        # Rust: TextGenerationItem { text: item.content }
        texts = [item['content'] for item in batch]
        
        # Tokenize
        # Rust: "[START]" + value + "[END]"
        # We can implement this manually or rely on tokenizer if we set bos/eos correctly.
        # Rust explicitly concatenates. Let's do same to be exact.
        processed_texts = [f"{self.tokenizer.bos_token}{text}{self.tokenizer.eos_token}" for text in texts]
        
        # Encoding with padding
        encodings = self.tokenizer(
            processed_texts, 
            padding="max_length", 
            truncation=True, 
            max_length=self.max_seq_length, 
            return_tensors="pt"
        )
        
        input_ids = encodings['input_ids'] # [batch, seq_len]
        attention_mask = encodings['attention_mask'] # [batch, seq_len], 1 for valid, 0 for pad
        
        # Create targets and inputs for Causal LM
        # Inputs: 0..-1
        # Targets: 1..end
        # Rust: 
        # inputs = tokens.slice([0..batch, 0..seq-1])
        # targets = tokens.slice([0..batch, 1..seq])
        
        # Note: input_ids is already max_seq_length.
        # We slice it.
        
        inputs = input_ids[:, :-1].clone()
        targets = input_ids[:, 1:].clone()
        
        # Mask pad: Rust uses Bool tensor.
        # mask_pad in Rust: True usually means "keep", but check batcher.rs
        # batcher.rs: generate_padding_mask.
        # In Burn, generate_padding_mask returns a mask where True usually denotes PADDING tokens (to be masked out).
        # Let's verify standard Burn behavior or Transformer behavior.
        # Actually Burn TransformerEncoder takes `mask_pad` where `True` positions are IGNORED.
        # PyTorch `src_key_padding_mask` also: `True` positions are IGNORED.
        # So we want `True` where it is padding.
        # tokenizer returns attention_mask where 1 is valid, 0 is pad.
        # So mask_pad (bool) should be (attention_mask == 0).
        
        mask_pad = (attention_mask[:, :-1] == 0) # Slice to match inputs length
        
        return {
            "inputs": inputs,
            "targets": targets,
            "mask_pad": mask_pad
        }
