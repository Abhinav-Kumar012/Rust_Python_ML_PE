from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
from transformers import BertTokenizer
import os
import contextlib

# Configuration (Must match training exactly)
class Config:
    d_model = 256
    nhead = 8
    num_layers = 4
    dim_feedforward = 1024
    num_classes = 4
    vocab_size = 28996
    max_seq_len = 256
    norm_first = True

# Model Definition (Must match training exactly)
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
            dropout=0.1,
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
        
        positions = torch.arange(0, seq_length, device=device).unsqueeze(0).repeat(batch_size, 1)
        
        embedded_tokens = self.embedding_token(tokens)
        embedded_positions = self.embedding_pos(positions)
        
        embedding = (embedded_positions + embedded_tokens) / 2
        
        src_key_padding_mask = mask_pad
        
        encoded = self.transformer(embedding, src_key_padding_mask=src_key_padding_mask)
        
        output = self.output(encoded)
        output_classification = output[:, 0, :]
        
        return output_classification

# Request/Response Schemas
class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    class_id: int
    class_name: str
    confidence: float

# Global State
ml_models = {}
tokenizer = None
device = None
CLASS_NAMES = {0: "World", 1: "Sports", 2: "Business", 3: "Technology"}

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # Load Model and Tokenizer
    global tokenizer, device
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model on {device}...")
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    
    config = Config()
    model = TextClassificationModel(config)
    
    model_path = "model_pytorch_text_classification/ag_news_model.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully.")
    else:
        print(f"WARNING: Model file not found at {model_path}. Inference will fail.")
    
    model.to(device)
    model.eval()
    
    ml_models["text_classifier"] = model
    
    yield
    
    # Cleanup
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if "text_classifier" not in ml_models:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    model = ml_models["text_classifier"]
    
    # Preprocessing
    encoding = tokenizer(
        [request.text],
        padding='max_length',
        truncation=True,
        max_length=Config.max_seq_len,
        return_tensors='pt',
        add_special_tokens=True
    )
    
    tokens = encoding['input_ids'].to(device)
    # Create mask: PyTorch Transformer expects True for padding (attention_mask is 0)
    mask_pad = (encoding['attention_mask'] == 0).to(device)
    
    with torch.no_grad():
        outputs = model(tokens, mask_pad)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)
    
    class_id = predicted_class.item()
    confidence_score = confidence.item()
    
    return PredictionResponse(
        class_id=class_id,
        class_name=CLASS_NAMES.get(class_id, "Unknown"),
        confidence=confidence_score
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
