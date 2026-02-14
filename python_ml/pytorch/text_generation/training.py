import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import sys
from tqdm import tqdm
import math

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import TextGenerationModel, TextGenerationModelConfig
from data import TextGenerationData, CollateFn

class NoamLR:
    def __init__(self, optimizer, model_size, warmup_steps, factor=1.0):
        self.optimizer = optimizer
        self.model_size = model_size
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.state_step = 0
    
    def step(self):
        self.state_step += 1
        # Noam formula: factor * (model_size ^ -0.5) * min(step ^ -0.5, step * warmup ^ -1.5)
        lr = self.factor * (self.model_size ** (-0.5) * min(self.state_step ** (-0.5), self.state_step * self.warmup_steps ** (-1.5)))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

def train(device, artifact_dir):
    # Ensure artifact directory exists
    os.makedirs(artifact_dir, exist_ok=True)
    
    # Configuration (matching Rust ExperimentConfig)
    config = TextGenerationModelConfig(
        vocab_size=0, # placeholder
        pad_token=0,  # placeholder
        max_seq_length=512,
        d_model=384,
        dim_feedforward=1536,
        nhead=12,
        num_layers=6,
        norm_first=True
    )
    
    batch_size = 6
    num_epochs = 50
    gradient_accumulation_steps = 6
    
    print("Loading data...")
    # Data Setup
    data_handler = TextGenerationData(max_seq_length=config.max_seq_length)
    train_dataset = data_handler.load_dbpedia("train")
    test_dataset = data_handler.load_dbpedia("test")
    
    # Update config
    config.vocab_size = data_handler.get_vocab_size()
    config.pad_token = data_handler.get_pad_token_id()
    
    print(f"Vocab size: {config.vocab_size}")
    
    collate_fn = CollateFn(data_handler.tokenizer, config.max_seq_length)
    
    # Rust uses num_workers=4
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn, 
        num_workers=0 # Set to 0 for Windows compatibility/debugging, or 4 if robust
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn, 
        num_workers=0
    )
    
    # Model Setup
    model = TextGenerationModel(config).to(device)
    
    # Optimizer (AdamW with weight decay 1e-6)
    # Rust: AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(1.0e-6)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0, weight_decay=1.0e-6)
    
    # Scheduler
    # Rust: NoamLrSchedulerConfig::new(0.01 / accum as f64).with_warmup_steps(6000)
    scheduler = NoamLR(
        optimizer, 
        model_size=config.d_model, 
        warmup_steps=6000, 
        factor=(0.01 / gradient_accumulation_steps)
    )
    
    print("Starting training...")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for step, batch in enumerate(progress_bar):
            inputs = batch['inputs'].to(device)
            targets = batch['targets'].to(device)
            mask_pad = batch['mask_pad'].to(device)
            
            outputs = model(inputs, mask_pad=mask_pad)
            loss = model.loss(outputs, targets)
            
            # Normalize loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            current_loss = loss.item() * gradient_accumulation_steps
            total_loss += current_loss
            
            # Update progress bar
            lr = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({"loss": f"{current_loss:.4f}", "lr": f"{lr:.6f}"})
        
        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Validation"):
                inputs = batch['inputs'].to(device)
                targets = batch['targets'].to(device)
                mask_pad = batch['mask_pad'].to(device)
                
                outputs = model(inputs, mask_pad=mask_pad)
                loss = model.loss(outputs, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(test_loader)
        print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}")
        
        # Checkpoint
        save_path = os.path.join(artifact_dir, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Saved checkpoint to {save_path}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create checkpoints directory
    artifact_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    
    train(device, artifact_dir)
