"""
Write from scratch a handful of torch.nn layers commonly used in ConvNets
and Transformers. In particular: 
-- Linear
-- Conv2d
-- Dropout
-- MaxPool
-- Avg Pool
-- LayerNorm
-- Multiheaded Attention
-- Tranformer Layer for a Decoder only network
Note!!! Conv2d and MaxPool/AvgPool layers currently use  torch.as_strided 
to perform the convolutions. While this somewhat works, it is
incredibly slow as written. Will fix later 
"""


import numpy as np
import torch.nn as nn
import torch,math
import mytorch.functions as mf



class Linear(nn.Module):
    """
    Applies a linear transformation to the input. 
    Args: 
    in_features: dimension of the input
    out_features: dimension of the output
    device/dtype
    """
    def __init__(self, in_features:int, out_features:int, device=None, dtype=None):
        super().__init__()
        k = 1/math.sqrt(in_features)
        self.weight = nn.Parameter(
            torch.empty(out_features,in_features, device = device, dtype=dtype).uniform_(-k,k))
        self.bias = nn.Parameter(torch.zeros(out_features, device=device,dtype=dtype))

    def forward(self,input):
        return input@self.weight.T + self.bias
    



class Conv2d(nn.Module):
    #TODO: Replace/modify as_strided call for faster convolutions
    """
    Applies a linear transformation to the input. 
    Args: 
    in_channels: number of input filters
    out_channels: number of output channels
    kernel_size: dimensions of the kernel, in tuple (kernel_height, kernel_width)
    stride: int
    padding: int
    device/dtype
    """
    def __init__(self, in_channels:int, out_channels:int, kernel_size:tuple, stride=1, padding=0,dtype=None, device=None):
        super().__init__()
        k = 1/math.sqrt(in_channels*kernel_size[0]*kernel_size[1])
        self.weight = nn.Parameter(torch.empty((out_channels,in_channels, *kernel_size),
                                               dtype=dtype, device=device).uniform_(-k,k)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels, dtype=dtype, device=device))
        self.padding=padding
        self.stride = stride
        self.kernel_size = kernel_size

        
       
    def forward(self,input):
        #input = tensor of size mb, in_c, H,W
        #each filter has size c_out, in_c, h,w
        kH,kW = self.kernel_size
        batch,in_c, iH, iW = input.shape
        oH,oW = int((iH+ 2*self.padding - kH)//self.stride + 1), int((iW+2*self.padding-kW)//self.stride + 1)
        if self.padding:
            inp_pad = torch.zeros(batch,in_c, iH+ 2*self.padding,
                                   iW+ 2*self.padding, dtype=input.dtype, device=input.device)
            inp_pad[:,:,self.padding:-self.padding, self.padding:-self.padding] = input
            input = inp_pad

        inp_strided = input.as_strided(size=(
            batch, in_c, oH,oW, kH,kW),
            stride=(input.stride(0), input.stride(1), 
            self.stride*input.stride(2),self.stride*input.stride(3),
            input.stride(2),input.stride(3)))
        output = torch.einsum('bihwkl,oikl->bohw', inp_strided, self.weight)
        return output
    
class Dropout(nn.Module):
    """
    Applies dropout with probability p. 
    Args: 
    p: probability
    """
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self,input):
        if not self.training or self.p==0:
            return input
        mask = torch.zeros(*input.shape, device=input.device).uniform_(0,1)>self.p
        return (1/(1-self.p))*mask*input

class MaxPool(nn.Module):
    #TODO: Replace/modify as_strided call for faster convolutions
    """
    Applies a 2d Max pooling over input. 
    Args: 
    kernel_size: as tuple (kernel_height, kernel_width)
    padding: int
    """
    def __init__(self,kernel_size, stride=1, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self,input):
        batch,in_c, iH, iW = input.shape
        kH, kW = self.kernel_size
        s = self.stride
        if self.padding:
            inp_pad = torch.zeros(batch,in_c, iH+ 2*self.padding,
                                   iW+ 2*self.padding, dtype=input.dtype, device=input.device)
            inp_pad[:,:,self.padding:-self.padding, self.padding:-self.padding] = input
            input = inp_pad
        oH  = (iH+ 2*self.padding - kH)//self.stride + 1
        oW = (iW+2*self.padding-kW)//self.stride + 1
        inp_strided = input.as_strided(size = (batch,in_c, oH,oW,kH,kW),
                                       stride = (input.stride(0), 
                                                 input.stride(1),
                                                 s*input.stride(2),
                                                 s*input.stride(3),
                                                 input.stride(2),
                                                 input.stride(3)))
        max_pool = inp_strided.amax(dim=(-1,-2))
        return max_pool
    
class AvgPool(nn.Module):
    #TODO: Replace/modify as_strided call for faster convolutions
    """
    Applies a 2d average pooling over input. 
    Args: 
    kernel_size: as tuple (kernel_height, kernel_width)
    padding: int
    """
    def __init__(self,kernel_size, stride=1, padding=0, dtype=None, device=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self,input):
        batch,in_c, iH, iW = input.shape
        kH, kW = self.kernel_size
        if self.padding:
            inp_pad = torch.zeros(batch,in_c, iH+ 2*self.padding,
                                   iW+ 2*self.padding, dtype=input.dtype)
            inp_pad[:,:,self.padding:-self.padding, self.padding:-self.padding] = input
            input = inp_pad
        oH, oW = int((iH+ 2*self.padding - kH)//self.stride + 1), int((iW+2*self.padding-kW)//self.stride + 1)
        inp_strided = input.as_strided(size = (batch,in_c, oH,oW,kH,kW),
                                       stride = (input.stride(0), input.stride(1),
                                                 self.stride*input.stride(2),
                                                 self.stride*input.stride(3),
                                                 input.stride(2),
                                                 input.stride(3)))
        print(inp_strided)
        avg_pool = torch.mean(inp_strided,dim=(-1,-2))
        return avg_pool


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0, device=None, dtype=None):
        super().__init__()
        """
        As detailed in Attention is All You Need. By default applies a causal mask to input
        Args: 
        embed_dim: int
        num_heads: int
        dropout: probability
        device/dtype
        """

        self.num_heads = num_heads
        self.head_dim = embed_dim//num_heads
        k_in = math.sqrt(3/(2*embed_dim)) #xavier initialization with in = 3*embed_dim, out =embed_dim
        k_out = math.sqrt(1/(embed_dim)) #initialized as linear
        self.in_weights = nn.Parameter(torch.empty(3,embed_dim,embed_dim, device=device, dtype=dtype).uniform_(-k_in,k_in))
        self.out_weights = nn.Parameter(torch.empty(embed_dim,embed_dim, device=device,dtype=dtype).uniform_(-k_out,k_out))
        self.dropout = Dropout(dropout)
        self.device=device
        self.dtype=dtype
    def forward(self, query, key,value, mask="causal"):
        """
        Input: q,k,v have shape (batch, sequence, embed_dim)
        Output: shape (batch, sequence, embed_dim)
        """
        batch, seq, embed_dim = query.shape
        query = (query@self.in_weights[0]).view(batch, seq, self.num_heads, self.head_dim).transpose(1,2)
        key = (key@self.in_weights[1]).view(batch, seq, self.num_heads, self.head_dim).transpose(1,2)
        value = (value@self.in_weights[2]).view(batch, seq, self.num_heads, self.head_dim).transpose(1,2)

        #q of size b,heads,seq,head_dim
        #k of size b,heads,seq,head_dim
        #contract over head_dim space, and scale by head_dim
        att_pat = query @ (key.transpose(-1,-2))/math.sqrt(self.head_dim)

        if mask=="causal":
            mask = torch.tril(torch.ones((seq, seq), device=self.device,dtype=self.dtype))
            att_pat = att_pat.masked_fill(mask==0, -float('inf'))
        att_pat = mf.softmax(att_pat,dim=-1)
        att_pat = self.dropout(att_pat)
        
        #(b,nh,T,T)x(b,nh,T,h_d)->(b,nh,T,h_d)
        att_head = att_pat @ value
        att_head = att_head.transpose(1,2).contiguous().view(batch,seq,embed_dim)
        #(batch, seq, embed_dim)
        att = self.dropout(att_head @ self.out_weights)
        return att #(batch, seq, embed_dim)
    

class LayerNorm(nn.Module):
    """
    Applies a layer normalization over neurons for each element in batch
    Args: 
    ndim: int
    eps: (regulator)
    bias: bool
    device/dtype
    """
    def __init__(self,ndim,eps=1e-05, bias=True,device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim,device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(ndim,device=device, dtype=dtype)) if bias else None
        self.eps = eps
    def forward(self, input):
        #input: (-1, ndim)
        mean = torch.mean(input,dim=-1, keepdim=True)
        var = torch.var(input, dim=-1, keepdim=True)
        input = ((input - mean)/(torch.sqrt(var + self.eps)))*self.weight + self.bias
        return input


class PositionalEncoder(nn.Module):
    """
    Performs a positional encoding for transformers, as in Attention is All you Need
    Args: 
    embed_dim: 
    lseq: max sequence length
    device/dtype
    """
    def __init__(self, embed_dim, lseq=5000,device=None, dtype=None):
        super().__init__()
        self.max_seq = lseq
        pos_enc = torch.zeros((lseq,embed_dim),device=device, dtype=dtype)
        pos = torch.arange(lseq,device=device, dtype=dtype).unsqueeze(1)
        i_emb = torch.arange(0,embed_dim,2,device=device, dtype=dtype)
        pos_enc[:,::2] = torch.sin(pos/(10000**(i_emb/embed_dim)))
        pos_enc[:,1::2] = torch.cos(pos/(10000**(i_emb/embed_dim)))[:,:embed_dim//2]
   
        self.register_buffer("pos_enc", pos_enc)
    def forward(self,input):
        lseq = input.shape[-2]
        if lseq>self.max_seq:
            raise ValueError("Input sequence exceeds max length")
        return input + self.pos_enc[:lseq].requires_grad_(False)

class Embedding(nn.Module):
    """
    Embeds input vector into embedding space
    Args: 
    vocab_size: int 
    embed_dim: int
    device/dtype
    """
    def __init__(self, vocab_size, embed_dim, device=None, dtype=None):
        super().__init__()
        self.embedding = nn.Parameter(torch.randn((vocab_size,embed_dim), device=device, dtype=dtype))
    def forward(self, input):
        return self.embedding[input]
    
class ReLU(nn.Module):
    """
    ReLU 
    """
    def __init__(self):
        super().__init__()
    def forward(self,input):
        return mf.relu(input)
    
class Flatten(nn.Module):
    """
    Flattens input from start dimension to end dimension
    """
    def __init__(self, start=0, end=-1):
        super().__init__()
        self.start = start
        self.end = end
    def forward(self,input):
        n = input.shape
        n0 = n[:self.start]
        if self.end==-1 or self.end==len(n)-1:
            return input.view(*n0, -1)
        n2 = n[self.end+1:]
        return input.view(*n0, -1, *n2)
       

class TransformerBlock(nn.Module):
    """
    Block for Decoder only transformer. Automatically includes causal mask for MHA layer
    Args: 
    embed_dim: embedding dimension
    num_head: number of heads
    d_fc: dimension of fully connected layers
    dropout: probability
    device/dtype
    """
    def __init__(self, embed_dim, num_heads, d_fc, dropout=0.0,device=None, dtype=None):
        super().__init__()
        self.attn = MultiheadAttention(embed_dim, num_heads, dropout=0.0,device=device, dtype=dtype)
        self.ln1 = LayerNorm(embed_dim, device=device, dtype=dtype)#after mha
        self.fc1 = Linear(embed_dim,d_fc,device=device, dtype=dtype)# for mlp, expanding from embedding space
        self.fc2 = Linear(d_fc, embed_dim,device=device, dtype=dtype)#for mlp, proj to embedding space
        self.ln2 = LayerNorm(embed_dim,device=device, dtype=dtype)#used after adding mha and mlp outputs
        self.do = Dropout(dropout) #used after mha before add and after mlp before add

    def forward(self, input):
        mh_out = self.ln1(input+ self.do(self.attn(input, input, input,mask="causal")))
        mlp_out = self.do(self.fc2(mf.relu(self.fc1(mh_out))))
        out = self.ln2(mlp_out + mh_out)
        return out
