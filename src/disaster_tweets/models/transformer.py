"""Member 3: encoder-only Transformer with bidirectional self-attention."""
import math
import torch
from torch import nn


class EncoderTransformer(nn.Module):
    def __init__(self,vocab_size,config,weights=None):
        super().__init__();self.config=config
        self.embedding=nn.Embedding(vocab_size,100,padding_idx=0)
        if weights is not None:
            with torch.no_grad():self.embedding.weight.copy_(torch.as_tensor(weights))
        self.embedding.weight.requires_grad_(not config.get('frozen',False))
        width=config['d_model'];self.projection=nn.Linear(100,width)
        position=torch.arange(64,dtype=torch.float32).unsqueeze(1)
        frequencies=torch.exp(torch.arange(0,width,2,dtype=torch.float32)*(-math.log(10000.)/width))
        encoding=torch.zeros(64,width);encoding[:,0::2]=torch.sin(position*frequencies);encoding[:,1::2]=torch.cos(position*frequencies)
        self.register_buffer('position_encoding',encoding.unsqueeze(0))
        self.input_norm=nn.LayerNorm(width)
        block=nn.TransformerEncoderLayer(width,config['heads'],dim_feedforward=config['feedforward'],dropout=config['dropout'],batch_first=True,norm_first=True,activation='gelu')
        self.encoder=nn.TransformerEncoder(block,num_layers=config['layers'],norm=nn.LayerNorm(width),enable_nested_tensor=False)
        # PyTorch clones initial layers; reset each layer independently for distinct parameters.
        for layer in self.encoder.layers:
            for p in layer.parameters():
                if p.dim()>1:nn.init.xavier_uniform_(p)
        self.head=nn.Sequential(nn.Dropout(config['dropout']),nn.Linear(2*width,64),nn.GELU(),nn.Dropout(config['dropout']),nn.Linear(64,1))

    def forward(self,tokens):
        lengths=tokens.ne(0).sum(1).clamp(min=1);width=int(lengths.max());tokens=tokens[:,:width]
        mask=torch.arange(width,device=tokens.device)[None,:]<lengths[:,None]
        embedded=self.projection(self.embedding(tokens))
        embedded=self.input_norm(embedded+self.position_encoding[:,:width])
        encoded=self.encoder(embedded,src_key_padding_mask=~mask)
        mean=(encoded*mask.unsqueeze(-1)).sum(1)/lengths[:,None]
        maximum=encoded.masked_fill(~mask.unsqueeze(-1),-torch.inf).amax(1)
        return self.head(torch.cat((mean,maximum),1)).squeeze(-1)
