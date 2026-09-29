import math
import torch
import torch.nn as nn

def scaled_dot_attention(Q,K,V,mask=None,return_attention=False):

    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)

    att = torch.softmax(scores, dim=-1)
    output = att@V

    if return_attention:
        return output, att
    return output

class MultiheadAttention(nn.Module):
    def __init__(self,d_model,n_head):
        super().__init__()
        if d_model % n_head != 0:
            print("d_model must be divisible by n_head")
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.w_q = nn.linear(d_model,d_model,bias=False)
        self.w_k = nn.linear(d_model,d_model,bias=False)
        self.w_v = nn.linear(d_model,d_model,bias=False)
        self.out_proj = nn.linear(d_model,d_model)
        self.attention_weight = None

    def forward(self,x,mask=None):
        B,T,_ = x.shape
        Q = self.w_q(x).view(B,T,self.n_head,self.d_k).transpose(1,2)
        K = self.w_k(x).view(B,T,self.n_head,self.d_k).transpose(1,2)
        V = self.w_v(x).view(B,T,self.n_head,self.d_k).transpose(1,2)

        out,att = scaled_dot_attention(Q,K,V,mask,return_attention=True)





