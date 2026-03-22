import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import TextGenerationModel, TextGenerationModelConfig
from data import TextGenerationData

app = FastAPI(title="Text Generation Inference")

class GenerateRequest(BaseModel):
    prompt: str
    max_length: int = 100
    temperature: float = 1.0

class GenerateResponse(BaseModel):
    generated_text: str

class TextGenerationInference:
    def __init__(self, model_path=None, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize Data Handler to get tokenizer and vocab size
        self.data_handler = TextGenerationData()
        self.tokenizer = self.data_handler.tokenizer
        
        # Define Special Tokens
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        
        # Load Model
        self.config = TextGenerationModelConfig(
            vocab_size=self.data_handler.get_vocab_size(),
            pad_token=self.pad_token_id,
            max_seq_length=512
        )
        self.model = TextGenerationModel(self.config).to(self.device)
        
        if model_path:
            if os.path.exists(model_path):
                print(f"Loading model from {model_path}")
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
            else:
                print(f"Warning: Model path {model_path} not found. Using initialized weights.")
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int = 100, temperature: float = 1.0):
        # Preprocessing: [START] + prompt
        input_text = f"{self.tokenizer.bos_token}{prompt}"
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)
        
        # Generation Loop
        curr_ids = input_ids
        
        for _ in range(max_new_tokens):
            # Forward pass
            # We need to handle seq length limit
            if curr_ids.shape[1] > self.config.max_seq_length:
                curr_ids = curr_ids[:, -self.config.max_seq_length:]
            
            with torch.no_grad():
                # mask_pad is None for inference usually, or we can create it if we had batching
                outputs = self.model(curr_ids)
                
            # Get logits of the last token
            next_token_logits = outputs[:, -1, :] # [batch, vocab]
            
            # Apply temperature
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            
            # Check for EOS
            if next_token.item() == self.eos_token_id:
                break
                
            # Append
            curr_ids = torch.cat([curr_ids, next_token], dim=1)
        
        # Decode
        generated_ids = curr_ids[0].tolist()
        # Skip special tokens? 
        decoded_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return decoded_text

# Global Inference Instance
inference_engine = None

@app.on_event("startup")
async def startup_event():
    global inference_engine
    # Look for the latest model checkpoint in the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Find files matching model_epoch_*.pth
    checkpoints = [f for f in os.listdir(current_dir) if f.startswith("model_epoch_") and f.endswith(".pth")]
    
    model_path = None
    if checkpoints:
        # Sort by epoch number
        # Filename format: model_epoch_{epoch}.pth
        try:
            checkpoints.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
            latest_checkpoint = checkpoints[-1]
            model_path = os.path.join(current_dir, latest_checkpoint)
        except Exception as e:
            print(f"Error sorting checkpoints: {e}")
            model_path = None
    
    inference_engine = TextGenerationInference(model_path=model_path)

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    if not inference_engine:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    try:
        result = inference_engine.generate(
            request.prompt, 
            max_new_tokens=request.max_length, 
            temperature=request.temperature
        )
        return GenerateResponse(generated_text=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
