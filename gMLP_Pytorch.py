import torch 
import torch.nn as nn 
import torch.nn.functional as F

class gMLPBLOCK(nn.Module):
    def __init__(self,d_model,d_ffn,seq_len):
        super(gMLPBLOCK,self).__init__()

        self.layer_norm = nn.LayerNorm(d_model)
        self.channel_proj_i = nn.Linear(d_model,d_ffn*2)
        self.channel_proj_ii = nn.Linear(d_ffn,d_model)
        self.sgu = SpatialGatingUnit(d_ffn,seq_len)

    def forward(self,x):
        residual = x
        x = self.layer_norm(x)
        x = F.gelu(self.channel_proj_i(x))
        x = self.sgu(x)
        x = self.channel_proj_ii(x)

        return residual + x

class SpatialGatingUnit(nn.Module):
    def __init__(self,d_ffn,seq_len):
        super(SpatialGatingUnit,self).__init__()

        self.layer_norm = nn.LayerNorm(d_ffn)
        self.spatial_proj = nn.Conv1d(seq_len,seq_len,1)
        nn.init.constant_(self.spatial_proj.bias, 1.0)

    def forward(self,x):
        u,v = x.chunk(2,dim=-1)
        v = self.layer_norm(v)
        v = self.spatial_proj(v)

        return u * v
