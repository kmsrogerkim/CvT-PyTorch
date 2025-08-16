import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    """
    MHA that accepts already-projected Q, K, V (with possibly different sequence lengths).
    """
    def __init__(self, dim: int, num_heads: int, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # final output projection
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        q: [B, Nq, D], k: [B, Nk, D], v: [B, Nk, D]
        returns: out [B, Nq, D]
        """
        B, Nq, D = q.shape
        _, Nk, _ = k.shape

        # reshape for heads
        def split_heads(t):
            return t.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, h, N, d]

        qh = split_heads(q)
        kh = split_heads(k)
        vh = split_heads(v)

        # scaled dot-product attention
        attn = (qh @ kh.transpose(-2, -1)) * self.scale        # [B, h, Nq, Nk]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ vh                                        # [B, h, Nq, d]

        # merge heads
        out = out.transpose(1, 2).contiguous().view(B, Nq, D)  # [B, Nq, D]
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class ConvTokenEmbedding(nn.Module):
    def __init__(self, in_ch, out_ch, patch_size, cls_token = False):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.k = patch_size
        self.s = patch_size
        self.p = patch_size//2

        # cls token is added only in stage 3
        if cls_token:
            # Learnable cls token: shape [1, 1, C]
            self.cls_token = nn.Parameter(torch.zeros(1, 1, out_ch))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.conv_layer = self.make_conv_layers() # [B, out_ch, H, W]
        self.layer_norm = nn.LayerNorm(out_ch)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layer(x)        

        x.flatten(2).transpose(1, 2)  # [B, N, D]
        x = self.layer_norm(x)
        
        B, _, _ = x.shape

        # cls token is added AFTER the embadding
        if self.cls_token is not None:
            cls = self.cls_token.expand(B, -1, -1) # [B, 1, C]
            x = torch.cat([cls, x], dim=1) # [B, 1+N, C]

        return x

    def make_conv_layers(self):
        return nn.Sequential(
            nn.Conv2d(self.in_ch, self.out_ch, self.k, self.s, self.p),
        )

class ConvTransformerBlock(nn.Module):
    # settings for stride for convolutional projection
    # is in Figure 3: (c) Squeezed convolutional projection
    def __init__(self, in_ch, dim, k = 3, s = 2,
                 num_heads = 8, attn_drop = 0.0, proj_drop = 0.0, mlp_ration = 4.0):
        super().__init__()

        self.dim = dim
        self.hidden_dim = int(mlp_ration * dim)
        self.mlp_drop = proj_drop

        # implementing "squeezed convolutional projection"
        # where the length for q is different from k & v
        self.q_dw_sperable_conv_layer = self.make_depth_wise_sperable_conv(in_ch, dim, k, s=1)
        self.k_dw_sperable_conv_layer = self.make_depth_wise_sperable_conv(in_ch, dim, k, s)
        self.v_dw_sperable_conv_layer = self.make_depth_wise_sperable_conv(in_ch, dim, k, s)

        self.multi_head_attention = MultiHeadAttention(dim, num_heads, attn_drop, proj_drop)

        self.mlp = self.make_mlp()

        self.layer_norm1 = nn.LayerNorm(dim)
        self.layer_norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convolutional projections
        q = self.q_dw_sperable_conv_layer(x)        # [B, D, Hq, Wq]
        k = self.k_dw_sperable_conv_layer(x)        # [B, D, Hk, Wk]
        v = self.v_dw_sperable_conv_layer(x)        # [B, D, Hk, Wk]

        B, D, Hq, Wq = q.shape

        # Flatten to sequences [B, N, D]
        def flatten(t: torch.Tensor) -> torch.Tensor:
            return t.flatten(2).transpose(1, 2)     # [B, N, D]

        q = flatten(q)
        k = flatten(k)
        v = flatten(v)

        # "flattened into size HiWi × Ci and normalized by layer normalization [1] 
        # for input into the subsequent Transformer blocks of stage i" (p. 4)

        q = self.layer_norm1(q)
        k = self.layer_norm1(k)
        v = self.layer_norm1(v)

        x = q + self.multi_head_attention(q, k, v)
        print(x.shape)
        x = x + self.mlp(self.layer_norm2(x))

        x = x.transpose(1, 2).contiguous().view(B, D, Hq, Wq)
        return x

    def make_depth_wise_sperable_conv(self, in_ch, out_ch, k, s):
        return nn.Sequential(
            # depth wise
            nn.Conv2d(in_ch, out_ch, k, s, padding=k//2, groups=in_ch),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),

            # point wise
            nn.Conv2d(out_ch, out_ch, kernel_size=1),
        )

    def make_mlp(self):
        return nn.Sequential(
            nn.Linear(self.dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.dim),
            nn.Dropout(self.mlp_drop)
        )

class CvT(nn.Module):
    def __init__(self):
        super().__init__()
        