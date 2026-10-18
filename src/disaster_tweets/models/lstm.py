"""Member 4: bidirectional LSTM with length-aware recurrent processing."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence


class LSTM(nn.Module):
    def __init__(self,vocab_size,config,weights=None):
        super().__init__();self.config=config
        self.embedding=nn.Embedding(vocab_size,100,padding_idx=0)
        if weights is not None:
            with torch.no_grad():self.embedding.weight.copy_(torch.as_tensor(weights))
        self.embedding.weight.requires_grad_(not config.get('frozen',False))
        self.input_dropout=nn.Dropout(.15)
        self.encoder=nn.LSTM(100,config['hidden'],num_layers=config.get('layers',1),batch_first=True,bidirectional=True,
                             dropout=config['dropout'] if config.get('layers',1)>1 else 0.)
        self.head=nn.Sequential(nn.LayerNorm(config['hidden']*4),nn.Dropout(config['dropout']),nn.Linear(config['hidden']*4,64),nn.ReLU(),nn.Dropout(config['dropout']),nn.Linear(64,1))

    def forward(self,tokens):
        lengths=tokens.ne(0).sum(1).clamp(min=1);width=int(lengths.max());tokens=tokens[:,:width]
        embedded=self.input_dropout(self.embedding(tokens))
        packed=pack_padded_sequence(embedded,lengths.cpu(),batch_first=True,enforce_sorted=False)
        encoded,_=self.encoder(packed);encoded,_=pad_packed_sequence(encoded,batch_first=True)
        mask=torch.arange(encoded.shape[1],device=tokens.device)[None,:]<lengths[:,None]
        mean=(encoded*mask.unsqueeze(-1)).sum(1)/lengths[:,None]
        maximum=encoded.masked_fill(~mask.unsqueeze(-1),-torch.inf).amax(1)
        return self.head(torch.cat((mean,maximum),1)).squeeze(-1)
