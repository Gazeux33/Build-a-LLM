from torch import nn
from torch.nn import functional as F
import torch

import config


class CasualSelfAttention(nn.Module):

    def __init__(self):
        super().__init__()
        assert config.N_EMBD % config.N_HEAD == 0
        self.c_attn = nn.Linear(config.N_EMBD, 3 * config.N_EMBD)
        self.c_proj = nn.Linear(config.N_EMBD, config.N_EMBD)

        self.n_head = config.N_HEAD
        self.n_embd = config.N_EMBD

        self.biais = torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size,
                                                                                       config.block_size)

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_fc = nn.Linear(config.N_EMBD, config.N_EMBD * 4)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(config.N_EMBD * 4, config.N_EMBD)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.N_EMBD)
        self.attn = CasualSelfAttention()
        self.ln_2 = nn.LayerNorm(config.N_EMBD)
        self.mlp = MLP()

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.VOCAB_SIZE, config.N_EMBD),
            wpe=nn.Embedding(config.block_size, config.N_EMBD),
            h=nn.ModuleList([Block() for _ in range(config.N_BLOCK)]),
            ln_f=nn.LayerNorm(config.N_EMBD)
        ))
        self.lm_head = nn.Linear(config.N_EMBD, config.VOCAB_SIZE, bias=False)

        # weight sharing
        self.transformer.wte.weight = self.lm_head.weight

        # init parms
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, target=None):
        B, T = idx.size()

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # (T)
        pos_emb = self.transformer.wpe(pos)  # (T, n_embd)
        tok_emb = self.transformer.wte(idx)  # (B, T, n_embd)

        x = tok_emb + pos_emb.unsqueeze(0)  # (B, T, n_embd)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        loss = None
        if target is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))  # (B*T,vocab_size) vs (B*T)
        return logits, loss

    @staticmethod
    def accuracy(logits, targets):
        _, predictions = torch.max(logits, dim=-1)
        correct_predictions = (predictions == targets).float()
        accuracy = correct_predictions.sum() / targets.numel()
        return accuracy
