import os
import random
import torch
import torch.nn as nn
from torch import Tensor
import timm
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.vision_transformer import Attention, Mlp, PatchEmbed, Block
from .positional_embedding import get_2d_sincos_pos_embed

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class AudioEncoder(nn.Module):
    def __init__(self, audio_length=1024, mel_bins=128, patch_size=16, embed_dim=768, mlp_ratio=4.,
                 num_heads=12, encoder_depth=12, qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        timm.models.vision_transformer.PatchEmbed = PatchEmbed
        timm.models.vision_transformer.Block = Block
        
        self.patch_embed = PatchEmbed(embed_dim=embed_dim, in_chans=1, patch_size=16, img_size=128)
        self.patch_embed.num_patches = int(audio_length * mel_bins / (patch_size ** 2))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim), requires_grad=True)
        self.modality = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.norm = nn.LayerNorm(embed_dim)
        
        self.transformer = nn.Sequential(*[
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=0.,
                norm_layer=nn.LayerNorm,
                act_layer=nn.GELU)
            for _ in range(encoder_depth)])
        
        self.initialize_weights()
    
    def forward(self, audio: Tensor) -> Tensor:
        """
        audio: (B, T, D) / (B, 1024, 128)
        """
        # Reshape input to (B, D, T) for Conv1d compatibility
        audio = audio.unsqueeze(1) # Shape: (B, 1, T, D)
        audio = audio.permute(0, 1, 3, 2)  # Shape: (B, 1, D, T)/(B, 1, 128, 1024)
        
        # Patchify the audio
        audio = self.patch_embed(audio)  # Shape: (B, embed_dim, new_T)
        
        # Add positional embeddings
        audio = audio + self.pos_embed
        audio = audio + self.modality
        
        for blk in self.transformer:
            audio = blk(audio)
        audio = self.norm(audio)
        
        return audio
    
    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], 8, int(self.patch_embed.num_patches / 8), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
        torch.nn.init.normal_(self.modality, std=.02)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
class AudioDecoder(nn.Module):
    def __init__(self, num_patches=(1024 * 128 // 256), patch_size=16, encoder_embed_dim=768, 
                 decoder_embed_dim=512, num_heads=16, decoder_depth=8, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        timm.models.vision_transformer.PatchEmbed = PatchEmbed
        timm.models.vision_transformer.Block = Block
        
        self.num_patches = num_patches
        
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        
        self.decoder_modality = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=True)
        
        self.blocks = nn.Sequential(*[
            Block(
                dim=decoder_embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=0.,
                norm_layer=nn.LayerNorm,
                act_layer=nn.GELU)
            for _ in range(decoder_depth)])
        
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * 1, bias=True)
        
        self.initialize_weights()
        
    def forward(self, audio):
        x = self.decoder_embed(audio)
        for blk in self.blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        
        return x
        
    def initialize_weights(self):
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], 8, int(self.num_patches/8), cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        
        torch.nn.init.normal_(self.decoder_modality, std=.02)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)