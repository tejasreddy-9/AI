from agno.knowledge.embedder.base import Embedder
from transformers import AutoTokenizer, AutoModel
import torch
from typing import Optional

class NomicAIEmbedder(Embedder):
    dimensions: Optional[int] = 768
    
    def __init__(self):
        self.id: str = 'nomic-ai/nomic-embed-text-v1.5'
   
    def get_embedding(self, text: str):
        """Get embedding for text and return as Python list."""
        tokenizer = AutoTokenizer.from_pretrained(self.id)
        model = AutoModel.from_pretrained(self.id, trust_remote_code=True)
        model.eval()
        
        encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        
        with torch.no_grad():
            model_output = model(**encoded_input)
        
        emb = model_output.last_hidden_state.mean(dim=1).squeeze()
        return emb.detach().cpu().numpy().tolist()
    
    def get_embedding_and_usage(self, text: str):
        """Get embedding with usage statistics."""
        tokenizer = AutoTokenizer.from_pretrained(self.id)
        model = AutoModel.from_pretrained(self.id, trust_remote_code=True)
        model.eval()
        
        encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        
        with torch.no_grad():
            model_output = model(**encoded_input)

        emb = model_output.last_hidden_state.mean(dim=1).squeeze()
        embedding = emb.detach().cpu().numpy().tolist()
        
        usage_data = {
            "tokens": len(encoded_input['input_ids'][0]),
            "input_length": len(text),
        }

        return embedding, usage_data