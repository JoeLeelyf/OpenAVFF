import torch
import torch.nn as nn
import torch.nn.functional as F
import einops


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

class SimpleTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SimpleTransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.fc2 = nn.Linear(embed_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        ffn_output = self.fc2(F.relu(self.fc1(x)))
        return self.norm2(x + ffn_output)

class A2VNetwork(nn.Module):
    def __init__(self, audio_dim, visual_dim, embed_dim, num_heads):
        super(A2VNetwork, self).__init__()
        self.mlp = MLP(audio_dim, visual_dim)
        self.transformer = SimpleTransformerBlock(embed_dim, num_heads)

    def forward(self, avis):
        b, t, c, d = avis.shape
        avis = einops.rearrange(avis, 'b t c d -> b d (t c)')
        va = self.mlp(avis)  # MLP to match dimensions
        va = va.transpose(1, 2)
        va = self.transformer(va)
        va = einops.rearrange(va, 'b (t c) d -> b t c d', t=t)
        return va

class V2ANetwork(nn.Module):
    def __init__(self, visual_dim, audio_dim, embed_dim, num_heads):
        super(V2ANetwork, self).__init__()
        self.mlp = MLP(visual_dim, audio_dim)
        self.transformer = SimpleTransformerBlock(embed_dim, num_heads)

    def forward(self, vvis):
        b, t, c, d = vvis.shape
        vvis = einops.rearrange(vvis, 'b t c d -> b d (t c)')
        av = self.mlp(vvis)  # MLP to match dimensions
        av = av.transpose(1, 2)
        av = self.transformer(av)
        av = einops.rearrange(av, 'b (t c) d -> b t c d', t=t)
        return av