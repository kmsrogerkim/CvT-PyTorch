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
    def __init__(self, in_ch, out_ch, k, s, add_cls_token = False, batch_size = 1):
        super().__init__()
        p = k//2
        self.conv_layer = nn.Conv2d(in_ch, out_ch, k, s, p)
        self.batch_norm = nn.BatchNorm2d(out_ch)

        self.add_cls_token = add_cls_token 
        if add_cls_token:
            cls_token = nn.Parameter(torch.zeros(1, 1, out_ch))
            self.cls_token = cls_token.expand(batch_size, -1, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.batch_norm(self.conv_layer(x))
        x = x.flatten(2).transpose(1, 2) # [B, N, D]
        
        if self.add_cls_token:
            x = torch.cat([x, self.cls_token], dim=1)
        return x

class ConvTransformerBlock(nn.Module):
    # settings for stride for convolutional projection
    # is in Figure 3: (c) Squeezed convolutional projection
    def __init__(self, in_ch, dim, k, s = 2,
                 num_heads = 8, attn_drop = 0.0, proj_drop = 0.0, mlp_ratio = 4.0):
        super().__init__()

        self.dim = dim
        self.hidden_dim = int(mlp_ratio * dim)
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

    def forward(self, x: torch.Tensor, cls_token = None) -> torch.Tensor:
        # Convolutional projections
        q = self.q_dw_sperable_conv_layer(x)        # [B, D, Hq, Wq]
        k = self.k_dw_sperable_conv_layer(x)        # [B, D, Hk, Wk]
        v = self.v_dw_sperable_conv_layer(x)        # [B, D, Hk, Wk]

        B, D, Hq, Wq = q.shape

        # Flatten to sequences [B, N, D]
        def flatten(t: torch.Tensor) -> torch.Tensor:
            return t.flatten(2).transpose(1, 2)

        q = flatten(q)
        k = flatten(k)
        v = flatten(v)

        q = self.layer_norm1(q)
        k = self.layer_norm1(k)
        v = self.layer_norm1(v)

        x = flatten(x)
        x = x + self.multi_head_attention(q, k, v)
        if cls_token is not None:
            x = torch.cat([cls_token, x], dim=1)
        x = x + self.mlp(self.layer_norm2(x))

        if cls_token is not None:
            cls_token = x[:, :1, :]        # [B, 1, D]
            x = x[:, 1:, :]

        x = x.transpose(1, 2).contiguous().view(B, D, Hq, Wq)
        return x, cls_token

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
    # For these configurations, go the Table 2 from the paper
    def __init__(self, batch_size, img_ch,
                depth1,depth2, depth3,
                # Conv Embadding parameters
                k1, c1, s1, k2, c2, s2, k3, c3, s3,
                # Conv Proj parameters
                kp1, kp2, kp3,
                # MHSA parameters
                H1, H2, H3,
                # MLP parameters
                R1, R2, R3, num_classes):
        super().__init__()
        # ----------------
        # Stage 1
        # ----------------
        self.embed1 = ConvTokenEmbedding(img_ch, c1, k1, s1, batch_size=batch_size)
        self.blocks1 = nn.ModuleList([
            ConvTransformerBlock(in_ch=c1, dim=c1, k=kp1, num_heads=H1, mlp_ratio=R1)
            for _ in range(depth1)
        ])

        # ----------------
        # Stage 2
        # ----------------
        self.embed2 = ConvTokenEmbedding(c1, c2, k2, s2, batch_size=batch_size)
        self.blocks2 = nn.ModuleList([
            ConvTransformerBlock(in_ch=c2, dim=c2, k=kp2, num_heads=H2, mlp_ratio=R2)
            for _ in range(depth2)
        ])

        # ----------------
        # Stage 3
        # ----------------
        self.embed3 = ConvTokenEmbedding(c2, c3, k3, s3, add_cls_token=True, batch_size=batch_size)
        self.blocks3 = nn.ModuleList([
            ConvTransformerBlock(in_ch=c3, dim=c3, k=kp3, num_heads=H3, mlp_ratio=R3)
            for _ in range(depth3)
        ])

        # final normalization + classifier head (use cls at the very end)
        self.head_norm = nn.LayerNorm(c3)
        self.head = nn.Linear(c3, num_classes)

    def forward(self, x: torch.Tensor):
        z1 = self.embed1(x)      # [B, N, D]
        # "flattened into size Hi*Wi × Ci and normalized by layer normalization [1] 
        # for input into the subsequent Transformer blocks of stage i" (p. 4).
        batch_size, n, c = z1.shape
        h = int(n**0.5)
        z1 = z1.reshape(batch_size, c, h, -1) # [B, D, H, W]
        for blk in self.blocks1:
            z1 = blk(z1)[0]         # shape stays 


        z2 = self.embed2(z1)
        batch_size, n, c = z2.shape
        h = int(n**0.5)
        z2 = z2.reshape(batch_size, c, h, -1)
        for blk in self.blocks2:
            z2 = blk(z2)[0]


        z3 = self.embed3(z2)
        cls, z3 = z3[:, :1, :], z3[:, 1:, :]  # split
        # reshape patch
        batch_size, n, c = z3.shape
        h = int(n**0.5)
        z3 = z3.reshape(batch_size, c, h, -1)
        for blk in self.blocks3:
            z3, cls = blk(z3, cls)         # [B, c3, H3, W3]

        # flatten grid tokens
        tokens = z3.flatten(2).transpose(1, 2)      # [B, N3, C3], N3 = H3*W3
        # concatenate cls token before final mlp layer
        tokens = torch.cat([cls, tokens], dim=1)# [B, 1+N3, C3]

        # final norm + take cls and classify
        tokens = self.head_norm(tokens)                    # LN over last dim
        cls_tok = tokens[:, 0]                             # [B, C3]
        logits = self.head(cls_tok)                        # [B, num_classes]
        return logits

model = CvT(batch_size=1, img_ch=3,
            depth1=1,depth2=2, depth3=10,
            # Conv Embadding parameters
            k1=7, c1=64, s1=4, k2=3, c2=192, s2=2, k3=3, c3=384, s3=2,
            # Conv Proj parameters
            kp1=3, kp2=3, kp3=3,
            # MHSA parameters
            H1=1, H2=3, H3=6,
            # MLP parameters
            R1=4, R2=4, R3=4, num_classes=1000)

x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    y = model(x)
print("logits shape:", y.shape)   # expected: [1, 1000]