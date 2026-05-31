import torch
import torch.nn.functional as F
from torch import nn
from timm.models.layers import drop, drop_path, trunc_normal_
from model.clip_model import QuickGELU
from torch.cuda.amp import autocast

#class Attention(nn.Module):
#    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
#        super().__init__()
#        self.num_heads = num_heads
#        head_dim = dim // num_heads
#        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
#        self.scale = qk_scale or head_dim ** -0.5
#
#        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
#        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
#        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
#
#
#        self.attn_drop = nn.Dropout(attn_drop)
#        self.proj = nn.Linear(dim, dim)
#        self.proj_drop = nn.Dropout(proj_drop)
#
#    def forward(self, q, k, v):
#        B, N, C = q.shape
#        assert k.shape == v.shape
#        B, M, C = k.shape
#        q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads)
#        k = self.k_proj(k).reshape(B, M, self.num_heads, C // self.num_heads)
#        v = self.v_proj(v).reshape(B, M, self.num_heads, C // self.num_heads)
#
#        attn = torch.einsum('bnkc,bmkc->bknm', q, k) * self.scale
#
#        attn = attn.softmax(dim=-1)
#
#        x = torch.einsum('bknm,bmkc->bnkc', attn, v).reshape(B, N, C)
#
#        x = self.proj(x)
#        x = self.proj_drop(x)
#        return x
#
#class TransformerDecoderLayer(nn.Module):
#    def __init__(
#        self,
#        d_model,
#        nhead,
#        dropout=0.1,
#    ):
#        super().__init__()
#        self.self_attn = Attention(d_model, nhead, proj_drop=dropout)
#        self.cross_attn = Attention(d_model, nhead, proj_drop=dropout)
#
#        self.norm1 = nn.LayerNorm(d_model)
#        self.norm2 = nn.LayerNorm(d_model)
#        self.norm3 = nn.LayerNorm(d_model)
#        self.dropout = nn.Dropout(dropout)
#
#        self.mlp = nn.Sequential(
#            nn.Linear(d_model, d_model * 4),
#            nn.GELU(),
#            nn.Dropout(dropout),
#            nn.Linear(d_model * 4, d_model)
#        )
#
#    def forward(self, x, mem):
#        q = k = v = self.norm1(x)
#        x = x + self.self_attn(q, k, v)
#        q = self.norm2(x)
#        x = x + self.cross_attn(q, mem, mem)
#        x = x + self.dropout(self.mlp(self.norm3(x)))
#        return x
#
#class ContextDecoder(nn.Module):
#    def __init__(self,
#                 transformer_width=256,
#                 transformer_heads=4,
#                 transformer_layers=6,
#                 visual_dim=512,
#                 dropout=0.1,
#                 **kwargs):
#        super().__init__()
#
#        self.memory_proj = nn.Sequential(
#            nn.LayerNorm(visual_dim),
#            nn.Linear(visual_dim, 256),
#            nn.LayerNorm(256),
#        )
#
#        self.text_proj = nn.Sequential(
#            nn.LayerNorm(visual_dim),
#            nn.Linear(visual_dim, 256),
#        )
#
#        self.decoder = nn.ModuleList([
#            TransformerDecoderLayer(256, transformer_heads, dropout) for _ in range(transformer_layers)
#        ])
#
#        self.out_proj = nn.Sequential(
#            nn.LayerNorm(256),
#            nn.Linear(256, visual_dim)
#        )
#
#        self.apply(self._init_weights)
#
#    def _init_weights(self, m):
#        if isinstance(m, nn.Linear):
#            trunc_normal_(m.weight, std=.02)
#            if isinstance(m, nn.Linear) and m.bias is not None:
#                nn.init.constant_(m.bias, 0)
#        elif isinstance(m, nn.LayerNorm):
#            nn.init.constant_(m.bias, 0)
#            nn.init.constant_(m.weight, 1.0)
#
#
#    def forward(self, text, visual):
#        #B, N, C = visual.shape
#        visual = self.memory_proj(visual)
#        x = self.text_proj(text)
#
#        for layer in self.decoder:
#            x = layer(x, visual)
#
#        return self.out_proj(x)

class MulitHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        B, N, C = q.shape
        B, M, C = k.shape
        q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(k).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(v).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PromptGeneratorLayer(nn.Module):
    def __init__(
            self,
            d_model,
            nhead,
            dropout=0.1,
    ):
        super().__init__()
        self.self_attn = MulitHeadAttention(d_model, nhead, proj_drop=dropout)
        self.cross_attn = MulitHeadAttention(d_model, nhead, proj_drop=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            QuickGELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x, visual):
        q = k = v = self.norm1(x)
        x = x + self.self_attn(q, k, v)
        q = self.norm2(x)
        x = x + self.cross_attn(q, visual, visual)
        x = x + self.dropout(self.mlp(self.norm3(x)))
        return x


class ImageSpecificPrompt(nn.Module):
    def __init__(self, layers=2, embed_dim=512, alpha=0.1, ):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)  # 512
        self.memory_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self.text_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )
        self.out_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )
        self.decoder = nn.ModuleList([PromptGeneratorLayer(embed_dim, embed_dim // 64) for _ in range(layers)])  # 2层
        self.alpha = nn.Parameter(torch.ones(embed_dim) * alpha)  # torch.Size([512])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, text, visual):  # text: torch.Size([64, 150, 512]) visual: torch.Size([64, 129, 512])
        # B, N, C = visual.shape
        with autocast():
            visual = self.memory_proj(visual)  # torch.Size([8, 129, 512])
            text = self.text_proj(text)  # torch.Size([8, 625, 512])
            # visual = self.norm(visual)  # torch.Size([4, 196, 512])  torch.Size([8, 129, 512])
            for layer in self.decoder:
                text = layer(text, visual)  # torch.Size([64, 150, 512])
            text = self.out_proj(text)
        return text

