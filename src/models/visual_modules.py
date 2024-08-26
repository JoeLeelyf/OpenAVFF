import torch
from einops import rearrange
from torch import nn, Tensor
from .utils import PatchEmbedding3d, Block3d, no_grad_trunc_normal_
from .positional_embedding import SinCosPositionalEmbedding

class VisualEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, n_frames=16, embed_dim=768, depth=12,
        num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
        norm_layer="LayerNorm", init_values=0., tubelet_size=2
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.patch_embedding = PatchEmbedding3d(
            input_size=(3, n_frames, img_size, img_size),
            patch_size=(tubelet_size, patch_size, patch_size),
            embedding=embed_dim
        )
        num_patches = (img_size // patch_size) * (img_size // patch_size) * (n_frames // tubelet_size)
        
        self.pos_embedding = SinCosPositionalEmbedding((num_patches, embed_dim), dropout_rate=0.)
        
        if norm_layer == 'LayerNorm':
            self.norm_layer = nn.LayerNorm
            self.norm = self.norm_layer(embed_dim)
        else:
            raise NotImplementedError(f"Normalization layer {norm_layer} not implemented")
        
        self.blocks = nn.ModuleList([
            Block3d(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=self.norm_layer,
                init_values=init_values)
            for _ in range(depth)
        ])
        
        self.apply(self._init_weights)
        
    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        # mask: (B, T, N) with boolean values, 0 -> masked, 1 -> visible
        assert len(x.shape) == 5, "x must be 5D"
        emb = self.patch_embedding(x)
        emb = self.pos_embedding(emb)
        emb = self.forward_features(emb)
        return emb

    def extract_features(self, x: Tensor, seq_mean_pool: bool) -> Tensor:
        x = self.patch_embedding(x)
        x = self.pos_embedding(x)
        for block in self.blocks:
            x = block(x)

        if seq_mean_pool:
            x = x.mean(dim=1)
        x = self.norm(x)
        return x
    
class VisualDecoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, n_frames=16, embed_dim=384, depth=8,
        num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
        norm_layer="LayerNorm", init_values=1., tubelet_size=2, encoder_embed_dim=768
    ):
        super().__init__()
        output_dim = 3 * tubelet_size * patch_size * patch_size
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.n_patch_h = img_size // patch_size
        self.n_patch_w = img_size // patch_size
        self.embed_dim = embed_dim
        if norm_layer == "LayerNorm":
            self.norm_layer = nn.LayerNorm
            self.norm = self.norm_layer(embed_dim)
        else:
            raise NotImplementedError("Only LayerNorm is supported")

        # sine-cosine positional embeddings
        self.pos_embedding = SinCosPositionalEmbedding(
            (self.n_patch_h * self.n_patch_w * (n_frames // tubelet_size), embed_dim), dropout_rate=0.)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = nn.ModuleList([
            Block3d(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=self.norm_layer,
                init_values=init_values
            ) for _ in range(depth)])

        self.head = nn.Linear(embed_dim, output_dim)
        self.apply(self._init_weights)
        no_grad_trunc_normal_(self.mask_token, mean=0., std=0.02, a=-0.02, b=0.02)
        
        self.decoder_patch_embed = nn.Linear(encoder_embed_dim, embed_dim)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def unpatch_to_img(self, x: Tensor) -> Tensor:
        # x: (Batch, No. batches, Prod of cube size * C)
        x = rearrange(x, "b n (c p) -> b n p c", c=3)
        # x: (Batch, No. batches, Prod of cube size, C)
        x = rearrange(x, "b (t h w) (p0 p1 p2) c -> b c (t p0) (h p1) (w p2)", p0=self.tubelet_size,
            p1=self.patch_size, p2=self.patch_size, h=self.n_patch_h, w=self.n_patch_w)
        # x: (B, C, T, H, W)
        return x

    def forward_features(self, x):
        for block in self.blocks:
            x = block(x)

        # if return_token_num > 0:
        #     x = x[:, -return_token_num:]

        x = self.norm(x)
        x = self.head(x)
        # x: (B, N_mask, C)
        return x

    def forward(self, x):
        # mask: 0 -> masked, 1 -> visible
        # b, n, c = x.shape
        # expand_pos_embed = self.pos_embedding.emb.data.expand(b, -1, -1)
        # pos_emb_vis = expand_pos_embed[mask].view(b, -1, c)
        # pos_emb_mask = expand_pos_embed[~mask].view(b, -1, c)
        # x = torch.cat([x + pos_emb_vis, self.mask_token + pos_emb_mask], dim=1)

        # mask_num = pos_emb_mask.shape[1]

        x = self.decoder_patch_embed(x)
        x = self.pos_embedding(x)
        x = self.forward_features(x)
        return x
