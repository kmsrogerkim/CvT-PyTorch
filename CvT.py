import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    """
    MHA that accepts already-projected Q, K, V.
    Supports different sequence lengths (Nq != Nk), masks, and SDPA fast path.
    """
    def __init__(self, dim: int, num_heads: int, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.attn_drop_p = float(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D] -> [B, h, N, d]
        B, N, D = x.shape
        h, d = self.num_heads, self.head_dim
        x = x.reshape(B, N, h, d).transpose(1, 2).contiguous()
        return x

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, h, N, d] -> [B, N, D]
        B, h, N, d = x.shape
        return x.transpose(1, 2).reshape(B, N, h * d)

    def forward(
        self,
        q: torch.Tensor,           # [B, Nq, D]
        k: torch.Tensor,           # [B, Nk, D]
        v: torch.Tensor,           # [B, Nk, D]
        attn_mask: torch.Tensor | None = None,        # broadcastable to [B*h, Nq, Nk] or [Nq, Nk]
        key_padding_mask: torch.Tensor | None = None  # [B, Nk], True for PAD positions
    ) -> torch.Tensor:
        B, Nq, D = q.shape
        assert k.shape[0] == B and v.shape[0] == B and k.shape[2] == D and v.shape[2] == D

        qh = self._split_heads(q)  # [B, h, Nq, d]
        kh = self._split_heads(k)  # [B, h, Nk, d]
        vh = self._split_heads(v)  # [B, h, Nk, d]

        # Build combined mask if key_padding_mask is provided
        # key_padding_mask: True means "mask out"
        if key_padding_mask is not None:
            # expand to [B, 1, 1, Nk] then broadcast to [B, h, Nq, Nk]
            kpm = key_padding_mask[:, None, None, :]  # bool
        else:
            kpm = None

        # Use PyTorch SDPA for speed/stability (handles scaling internally if you pass scaled q)
        # We pass scaled q ourselves to match your original scale factor.
        qh_scaled = qh * self.scale

        # SDPA expects masks float/bool: attn_mask can be either additive (float) or boolean.
        # If both masks exist, combine them into a single boolean mask.
        combined_mask = None
        if (attn_mask is not None) and (kpm is not None):
            # convert attn_mask to boolean if needed
            am = attn_mask
            if am.dtype != torch.bool:
                # treat -inf / large negative as masked
                am = am == float("-inf")
            combined_mask = am | kpm
        elif attn_mask is not None:
            combined_mask = attn_mask
        else:
            combined_mask = kpm

        out = F.scaled_dot_product_attention(
            qh_scaled, kh, vh,
            attn_mask=combined_mask,
            dropout_p=self.attn_drop_p if self.training else 0.0,
            is_causal=False
        )  # [B, h, Nq, d]

        out = self._merge_heads(out)    # [B, Nq, D]
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class ConvTokenEmbedding(nn.Module):
    def __init__(self, in_ch, out_ch, k, s, add_cls_token = False):
        super().__init__()
        p = k//2
        self.conv_layer = nn.Conv2d(in_ch, out_ch, k, s, p)
        self.layer_norm = nn.LayerNorm(out_ch)

        self.add_cls_token = add_cls_token 
        if add_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layer(x)
        x = x.flatten(2).transpose(1, 2) # [B, N, D]
        if self.add_cls_token:
            cls = self.cls_token.expand(x.size(0), -1, -1).to(device=x.device, dtype=x.dtype)
            x = torch.cat([x, cls], dim=1)
        x = self.layer_norm(x)
        return x

class ConvTransformerBlock(nn.Module):
    # settings for stride for convolutional projection
    # is in Figure 3: (c) Squeezed convolutional projection
    def __init__(self, in_ch, dim, k, s = 2,
                 num_heads = 8, attn_drop = 0.0, proj_drop = 0.0, mlp_ratio = 4.0, mlp_drop = 0.0):
        super().__init__()

        self.dim = dim
        self.hidden_dim = int(mlp_ratio * dim)
        self.mlp_drop = mlp_drop 

        # implementing "squeezed convolutional projection"
        # where the length for q is different from k & v
        self.q_dw_separable_conv_layer = self.make_depth_wise_sperable_conv(in_ch, dim, k, s=1)
        self.k_dw_separable_conv_layer = self.make_depth_wise_sperable_conv(in_ch, dim, k, s)
        self.v_dw_separable_conv_layer = self.make_depth_wise_sperable_conv(in_ch, dim, k, s)

        self.multi_head_attention = MultiHeadAttention(dim, num_heads, attn_drop, proj_drop)

        self.mlp = self.make_mlp()

        self.pre_norm = nn.LayerNorm(dim)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, cls_token = None) -> torch.Tensor:
        # Flatten to sequences [B, N, D]
        def flatten(t: torch.Tensor) -> torch.Tensor:
            return t.flatten(2).transpose(1, 2)

        # pre-norm
        B, D, H, W = x.shape
        x = flatten(x)
        if cls_token is not None:
            x = torch.cat([cls_token, x], dim=1)
        x = self.pre_norm(x)
        if cls_token is not None:
            cls_token, x = x[:, :1, :], x[:, 1:, :]
        x = x.reshape(B, D, H, W)

        # Convolutional projections (spatial tokens only, no cls token)
        q = self.q_dw_separable_conv_layer(x)        # [B, D, Hq, Wq]
        k = self.k_dw_separable_conv_layer(x)        # [B, D, Hk, Wk]
        v = self.v_dw_separable_conv_layer(x)        # [B, D, Hk, Wk]

        B, D, Hq, Wq = q.shape

        q = flatten(q)
        k = flatten(k)
        v = flatten(v)

        x = flatten(x)
        if cls_token is not None:
            # re-attach the cls token before attention
            x = torch.cat([cls_token, x], dim=1)
            q = torch.cat([cls_token, q], dim=1)
            k = torch.cat([cls_token, k], dim=1)
            v = torch.cat([cls_token, v], dim=1)
        x = x + self.multi_head_attention(q, k, v)
        x = x + self.mlp(self.layer_norm(x))

        if cls_token is not None:
            cls_token = x[:, :1, :]
            x = x[:, 1:, :]

        x = x.transpose(1, 2).reshape(B, D, Hq, Wq)
        return x, cls_token

    def make_depth_wise_sperable_conv(self, in_ch, out_ch, k, s):
        return nn.Sequential(
            nn.Conv2d(in_ch, in_ch, k, s, padding=k//2, groups=in_ch),
            nn.BatchNorm2d(in_ch),
            nn.GELU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=1),
            nn.GELU(),
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
    def __init__(self, img_ch,
                # dropout rates
                attn_drop, proj_drop, mlp_drop,
                # Depth of stage
                depth1, depth2, depth3,
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
        self.embed1 = ConvTokenEmbedding(img_ch, c1, k1, s1)
        self.blocks1 = nn.ModuleList([
            ConvTransformerBlock(in_ch=c1, dim=c1, k=kp1, num_heads=H1, mlp_ratio=R1, 
                                 attn_drop=attn_drop, proj_drop=proj_drop, mlp_drop=mlp_drop)
            for _ in range(depth1)
        ])

        # ----------------
        # Stage 2
        # ----------------
        self.embed2 = ConvTokenEmbedding(c1, c2, k2, s2)
        self.blocks2 = nn.ModuleList([
            ConvTransformerBlock(in_ch=c2, dim=c2, k=kp2, num_heads=H2, mlp_ratio=R2,
                                 attn_drop=attn_drop, proj_drop=proj_drop, mlp_drop=mlp_drop)
            for _ in range(depth2)
        ])

        # ----------------
        # Stage 3
        # ----------------
        self.embed3 = ConvTokenEmbedding(c2, c3, k3, s3, add_cls_token=True)
        self.blocks3 = nn.ModuleList([
            ConvTransformerBlock(in_ch=c3, dim=c3, k=kp3, num_heads=H3, mlp_ratio=R3,
                                 attn_drop=attn_drop, proj_drop=proj_drop, mlp_drop=mlp_drop)
            for _ in range(depth3)
        ])

        # mlp head
        self.mlp_head = nn.Linear(c3, num_classes)

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

        # ----------------
        # Stage 3
        # ----------------
        z3 = self.embed3(z2)
        cls, z3 = z3[:, :1, :], z3[:, 1:, :]  # split cls token from spatial patch

        # reshape patch
        batch_size, n, c = z3.shape
        h = int(n**0.5)
        z3 = z3.reshape(batch_size, c, h, -1)
        for blk in self.blocks3:
            z3, cls = blk(z3, cls)
            # z3: [B, C3, H, W]
            # cls: [B, 1, C3]

        # mlp head
        cls = cls.squeeze(1)
        return self.mlp_head(cls)

model = CvT(img_ch=3,
            attn_drop=0.1, proj_drop=0.1, mlp_drop=0.1,
            depth1=1, depth2=2, depth3=10,
            # Conv Embadding parameters
            k1=7, c1=64, s1=4, k2=3, c2=192, s2=2, k3=3, c3=384, s3=2,
            # Conv Proj parameters
            kp1=3, kp2=3, kp3=3,
            # MHSA parameters
            H1=1, H2=3, H3=6,
            # MLP parameters
            R1=4, R2=4, R3=4, num_classes=37)

x = torch.rand([6, 3, 224, 224])
x = x.to(torch.device("cuda"))
model = model.to(torch.device("cuda"))
print(model(x).shape)