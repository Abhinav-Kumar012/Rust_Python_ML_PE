import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from datasets import load_dataset
import math
from tqdm import tqdm
import os

# Configuration
class Config:
    d_model = 256
    nhead = 8
    num_layers = 4
    dim_feedforward = 1024
    num_classes = 4  # AG News has 4 classes
    vocab_size = 28996 # bert-base-cased vocab size
    max_seq_len = 256
    batch_size = 8
    num_epochs = 5
    lr = 1e-2
    warmup_steps = 1000
    weight_decay = 5e-5
    norm_first = True
    quiet_softmax = True # Burn specific, PyTorch standard softmax is fine

class TextClassificationModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        
        # Embeddings
        self.embedding_token = nn.Embedding(config.vocab_size, config.d_model)
        self.embedding_pos = nn.Embedding(config.max_seq_len, config.d_model)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            batch_first=True,
            norm_first=config.norm_first
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=config.num_layers,
            norm=nn.LayerNorm(config.d_model) if config.norm_first else None
        )
        
        # Output
        self.output = nn.Linear(config.d_model, config.num_classes)
        
        self.max_seq_len = config.max_seq_len

    def forward(self, tokens, mask_pad=None):
        batch_size, seq_length = tokens.shape
        device = tokens.device
        
        # Positions: [0, 1, 2, ..., seq_length-1] repeated for batch
        positions = torch.arange(0, seq_length, device=device).unsqueeze(0).repeat(batch_size, 1)
        
        embedded_tokens = self.embedding_token(tokens)
        embedded_positions = self.embedding_pos(positions)
        
        # Burn implementation: (pos + tok) / 2
        embedding = (embedded_positions + embedded_tokens) / 2
        
        # Mask padding for transformer
        # Burn mask_pad: true for padding ?
        # PyTorch src_key_padding_mask: True for values to be ignored (padded)
        # We need to ensure the mask logic matches. usually 0 is pad in BERT.
        
        src_key_padding_mask = None
        if mask_pad is not None:
            # Check mask convention. If mask_pad is BoolTensor where True = padding
            src_key_padding_mask = mask_pad
            
        encoded = self.transformer(embedding, src_key_padding_mask=src_key_padding_mask)
        
        # Slice first token (CLS equivalent in position 0)
        # Rust: output.slice([0..batch_size, 0..1]).reshape([batch_size, n_classes]) 
        # Wait, Rust slice is on the output of Linear?
        # Rust:
        # let encoded = self.transformer.forward(...)
        # let output = self.output.forward(encoded);
        # let output_classification = output.slice([0..batch_size, 0..1]).reshape(...)
        
        output = self.output(encoded) # [batch, seq, num_classes]
        
        # Slice [:, 0, :] -> [batch, num_classes]
        output_classification = output[:, 0, :]
        
        return output_classification

class NoamLR(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, model_size, warmup_steps, factor=1.0, last_epoch=-1):
        self.model_size = model_size
        self.warmup_steps = warmup_steps
        self.factor = factor
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(1, self.last_epoch + 1)
        scale = self.factor * (self.model_size ** (-0.5)) * min(step ** (-0.5), step * self.warmup_steps ** (-1.5))
        return [scale for _ in self.base_lrs]

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
def collate_fn(batch):
    # Batch is list of dicts: {'text': ..., 'label': ...}
    
    texts = [item['text'] for item in batch]
    labels = [item['label'] for item in batch]
    
    # Tokenize
    encoding = tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=Config.max_seq_len,
        return_tensors='pt',
        add_special_tokens=True # Should imply CLS/SEP
    )
    
    tokens = encoding['input_ids']
    # Create mask: PyTorch Transformer expects True for padding
    # encoding['attention_mask'] is 1 for real, 0 for pad.
    # We want True for pad (0), False for real (1).
    mask_pad = (encoding['attention_mask'] == 0)
    
    labels = torch.tensor(labels, dtype=torch.long)
    
    return tokens, labels, mask_pad

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data
    print("Loading dataset...")
    # Using 'ag_news' from huggingface datasets
    dataset = load_dataset(
        "ag_news",
        trust_remote_code=False,
        download_mode="reuse_cache_if_exists",
        verification_mode="no_checks",
    )

    
    # Rust used a subset? "SamplerDataset::new(dataset_train, 50_000)"
    # ag_news train has 120k. Rust used 50k.
    train_dataset = dataset['train'].shuffle(seed=42).select(range(50000))
    # Rust used 5k for test. ag_news test has 7.6k.
    test_dataset = dataset['test'].shuffle(seed=42).select(range(5000))
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=4 # Simple for now
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=Config.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=4
    )
    
    # Model
    config = Config()
    model = TextClassificationModel(config).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), weight_decay=config.weight_decay)
    
    # Scheduler
    # Rust: NoamLrSchedulerConfig::new(1e-2).with_warmup_steps(1000).with_model_size(...)
    scheduler = NoamLR(optimizer, config.d_model, config.warmup_steps, factor=config.lr)
    
    # Loss
    criterion = nn.CrossEntropyLoss()
    
    # Training Loop
    print("Starting training...")
    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Train]")
        
        for tokens, labels, mask_pad in progress_bar:
            tokens, labels, mask_pad = tokens.to(device), labels.to(device), mask_pad.to(device)
            
            optimizer.zero_grad()
            outputs = model(tokens, mask_pad)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item() * tokens.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix({'loss': loss.item()})
            
        avg_train_loss = train_loss / total_train
        train_acc = train_correct / total_train
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val = 0
        
        with torch.no_grad():
            for tokens, labels, mask_pad in tqdm(test_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Val]"):
                tokens, labels, mask_pad = tokens.to(device), labels.to(device), mask_pad.to(device)
                
                outputs = model(tokens, mask_pad)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * tokens.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        avg_val_loss = val_loss / total_val
        val_acc = val_correct / total_val
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
              
    # Save model
    os.makedirs("model_pytorch_text_classification", exist_ok=True)
    torch.save(model.state_dict(), "model_pytorch_text_classification/ag_news_model.pth")
    print("Model saved to model_pytorch_text_classification/ag_news_model.pth")

if __name__ == "__main__":
    train()
