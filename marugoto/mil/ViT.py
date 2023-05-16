
import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn

from .transformer import PreNorm, Attention, FeedForward

# stolen from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_vit.py
def posemb_sincos_2d(x, coords, temperature = 10000, dtype = torch.float32):
    b,dim, device, dtype = x.shape[0],x.shape[-1], x.device, x.dtype
    #y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
    y, x = coords[:,:,1], coords[:,:,0]
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)   
    y = y.view(b,-1)[:,:, None] * omega[None, :]
    x = x.view(b,-1)[:,:, None] * omega[None, :] 
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 2)/10
    return pe.type(dtype)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x): #, register_hook=False
        for attn, ff in self.layers:
            x = attn(x) + x # , register_hook=register_hook
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(self, *, num_classes, input_dim=768, dim=512, depth=2, heads=8, mlp_dim=512, pool='cls', channels=3,
                 dim_head=64, dropout=0., emb_dropout=0., nr_tiles=4096, add_pos_feats=False):
        super().__init__()
        # image_height, image_width = pair(image_size)
        # patch_height, patch_width = pair(patch_size)
        #
        # assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        #
        # num_patches = (image_height // patch_height) * (image_width // patch_width)
        # patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # self.to_patch_embedding = nn.Sequential(
        #     Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
        #     nn.Linear(patch_dim, dim),
        # )

        self.pos_embedding = nn.Parameter(torch.randn(1, nr_tiles+1, dim))
        
        self.input_dim = input_dim
        
        self.omega_x = nn.Parameter(torch.arange(dim // 4) / (dim // 4 - 1))
        self.omega_y = nn.Parameter(torch.arange(dim // 4) / (dim // 4 - 1))
        
        self.scale = nn.Parameter(torch.randn(dim))
        
        if(add_pos_feats):
            self.input_dim=self.input_dim + 2     
        self.fc = nn.Sequential(nn.Linear(self.input_dim, dim, bias=True), nn.ReLU())  # added by me

        #self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        #self.transformer = Transformer(self.input_dim, depth, heads, dim_head, mlp_dim, dropout)
        
        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(self.input_dim),
        #     nn.Linear(self.input_dim, num_classes)
        # )
    
    def learnable_sincos_2d(self, x, coords, omega_x, omega_y, temperature = 10000, dtype = torch.float32):
        b = x.shape[0]
        #y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
        y, x = coords[:,:,1], coords[:,:,0]
        assert (self.input_dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
        #omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
        adj_omega_y = 1. / (temperature ** omega_y)
        adj_omega_x = 1. / (temperature ** omega_x)   
        y = y.view(b,-1)[:,:, None] * adj_omega_y[None, :]
        x = x.view(b,-1)[:,:, None] * adj_omega_x[None, :] 
        pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 2)/10
        return pe.type(dtype)

    def forward(self, x, coords, register_hook=False):
        # x = self.to_patch_embedding(img)
        b, n, d = x.shape
        #print(f"input shape: {x.shape}")
        #pe = posemb_sincos_2d(x,coords)
        #pe = self.learnable_sincos_2d(x,coords,self.omega_x,self.omega_y)
        #x = x + pe
        #pe = posemb_sincos_2d(x,coords)
        #x = x + pe
        #x = torch.cat((x,coords/3e+4),dim=2)
        x = self.fc(x)
        #pe = self.learnable_sincos_2d(x,coords,self.omega_x,self.omega_y)
        
        #pe = self.learnable_sincos_2d(x,coords,self.omega_x,self.omega_y)
        #x = x + pe*self.scale[:n]
        
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        #pe = self.learnable_sincos_2d(x,coords,self.omega_x,self.omega_y)
        
        #x = x + self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x) # , register_hook=register_hook

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
