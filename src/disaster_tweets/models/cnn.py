"""Member 1: parallel convolutional phrase classifier."""
import torch
from torch import nn


class CNN(nn.Module):
    def __init__(self,vocab_size,config,weights=None):
        super().__init__();self.config=config
        self.embedding=nn.Embedding(vocab_size,100,padding_idx=0)
        if weights is not None:
            with torch.no_grad():self.embedding.weight.copy_(torch.as_tensor(weights))
        self.embedding.weight.requires_grad_(not config.get('frozen',False))
        self.embedding_dropout=nn.Dropout1d(.1)
        self.convs=nn.ModuleList([nn.Conv1d(100,config['filters'],k) for k in config['kernels']])
        self.head=nn.Sequential(nn.Dropout(config['dropout']),nn.Linear(config['filters']*len(config['kernels']),1))

    def forward(self,tokens):
        lengths=tokens.ne(0).sum(1);width=max(max(self.config['kernels']),int(lengths.max()))
        tokens=tokens[:,:width]
        if tokens.shape[1]<width:tokens=nn.functional.pad(tokens,(0,width-tokens.shape[1]))
        embedded=self.embedding_dropout(self.embedding(tokens).transpose(1,2));pooled=[]
        for kernel,conv in zip(self.config['kernels'],self.convs):
            features=torch.relu(conv(embedded));valid=(lengths-kernel+1).clamp(min=1)
            mask=torch.arange(features.shape[2],device=tokens.device)[None,:]<valid[:,None]
            pooled.append(features.masked_fill(~mask[:,None,:],-torch.inf).amax(2))
        return self.head(torch.cat(pooled,1)).squeeze(-1)
