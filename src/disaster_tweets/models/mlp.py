"""Member 2: pooled-embedding multilayer perceptron."""
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self,vocab_size,config,weights=None):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,100,padding_idx=0)
        if weights is not None:
            with torch.no_grad():self.embedding.weight.copy_(torch.as_tensor(weights))
        self.embedding.weight.requires_grad_(not config.get('frozen',False))
        self.head=nn.Sequential(nn.LayerNorm(200),nn.Dropout(config['dropout']),nn.Linear(200,config['hidden']),nn.ReLU(),nn.Dropout(config['dropout']),nn.Linear(config['hidden'],1))

    def forward(self,tokens):
        mask=tokens.ne(0).unsqueeze(-1);embedded=self.embedding(tokens)
        mean=(embedded*mask).sum(1)/mask.sum(1).clamp(min=1)
        maximum=embedded.masked_fill(~mask,-torch.inf).amax(1)
        maximum=torch.where(torch.isfinite(maximum),maximum,torch.zeros_like(maximum))
        return self.head(torch.cat((mean,maximum),1)).squeeze(-1)
