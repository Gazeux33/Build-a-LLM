from tokenizers import Tokenizer
import os
import json
from torch import nn
from torch.nn import functional as F

from config import *


def get_tokenizer(path: str) -> Tokenizer:
    print(path)
    return Tokenizer.from_file(path)


def get_tokens_paths(tokens_dir: str, split: str) -> list[str]:
    return sorted([os.path.join(tokens_dir, f) for f in (os.listdir(tokens_dir)) if f.endswith(".npy") and split in f])


def get_vocab() -> dict:
    with open(VOCAB_PATH, "r") as f:
        data = json.load(f)
    return data["model"]["vocab"]


def save_model(model: nn.Module, name: str) -> None:
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, name))


def load_weights(model: nn.Module, name: str) -> nn.Module:
    path = os.path.join(MODEL_DIR, name)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
    return model

def generate(model:nn.Module,idx:torch.Tensor,max_token_gen:int):
    model.eval()
    with torch.no_grad():
        for i in range(max_token_gen):
            logits,_ = model(idx)
            logits = logits[:,-1,:]
            probs = F.softmax(logits,dim=-1)
            idx_next = torch.multinomial(probs,num_samples=1)
            idx = torch.cat([idx,idx_next],dim=1)
    return idx

def tokenize_text(tokenizer:Tokenizer,text:str)->torch.Tensor:
    return torch.tensor(tokenizer.encode(text).ids, dtype=torch.long).unsqueeze(0)

def detokenize_text(tokenizer:Tokenizer,idx:torch.Tensor)->str:
    return tokenizer.decode(idx.squeeze(0).tolist())
