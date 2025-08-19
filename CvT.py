import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.out_proj = nn.Linear(dim, dim, bias=True)  # W_o
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):  # [B,N,D] each
        B, Nq, D = q.shape
        h, d = self.num_heads, self.head_dim
        q = q.view(B, Nq, h, d).transpose(1, 2)
        k = k.view(B, -1, h, d).transpose(1, 2)
        v = v.view(B, -1, h, d).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        out = attn @ v                             # [B,h,Nq,d]
        out = out.transpose(1, 2).reshape(B, Nq, D)
        out = self.out_proj(out)                   # output projection
        out = self.proj_drop(out)                  # <- proj_drop lives here
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
        # "flattened into size Hi*Wi × Ci and normalized by layer normalization
        # for input into the subsequent Transformer blocks of stage i" (p. 4).
        return x

class ConvTransformerBlock(nn.Module):
    # settings for stride for convolutional projection
    # is in Figure 3: (c) Squeezed convolutional projection
    def __init__(self, in_ch, dim, k, s = 2,
                 num_heads = 8, attn_drop = 0.0, proj_drop = 0.0, mlp_ratio = 4.0):
        super().__init__()

        self.dim = dim
        self.hidden_dim = int(mlp_ratio * dim)
        self.proj_drop = proj_drop

        # implementing "squeezed convolutional projection"
        # where the length for q is different from k & v
        self.q_dw_separable_conv_layer = self.make_depth_wise_sperable_conv(in_ch, dim, k, s=1)
        self.k_dw_separable_conv_layer = self.make_depth_wise_sperable_conv(in_ch, dim, k, s)
        self.v_dw_separable_conv_layer = self.make_depth_wise_sperable_conv(in_ch, dim, k, s)

        # self.multi_head_attention = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        self.multi_head_attention = MultiHeadAttention(dim, num_heads, attn_drop=attn_drop, proj_drop=proj_drop)

        self.mlp = self.make_mlp()

        self.pre_norm = nn.LayerNorm(dim)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, cls_token = None) -> torch.Tensor:
        # Flatten to sequences [B, N, D]
        def flatten(t: torch.Tensor) -> torch.Tensor:
            return t.flatten(2).transpose(1, 2)

        if x.dim() == 3:
            batch_size, n, d = x.shape
            h = int(n**0.5)
            x = x.reshape(batch_size, d, h, h) # [B, D, H, W]

        # Convolutional projections (spatial tokens only, no cls token)
        q = self.q_dw_separable_conv_layer(x)        # [B, D, Hq, Wq]
        k = self.k_dw_separable_conv_layer(x)        # [B, D, Hk, Wk]
        v = self.v_dw_separable_conv_layer(x)        # [B, D, Hk, Wk]

        B, D, H, W = q.shape

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
        x = x + self.multi_head_attention(q, k, v)[0]
        x = x + self.mlp(self.layer_norm(x))    # [B, 1+N, D]

        if cls_token is not None:
            cls_token = x[:, :1, :]
            x = x[:, 1:, :]

        x = x.transpose(1, 2).reshape(B, D, H, W)
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
            nn.Dropout(self.proj_drop)
        )

class CvT(nn.Module):
    # For these configurations, go the Table 2 from the paper
    def __init__(self, img_ch,
                # dropout rates
                attn_drop, proj_drop,
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
                                 attn_drop=attn_drop, proj_drop=proj_drop)
            for _ in range(depth1)
        ])

        # ----------------
        # Stage 2
        # ----------------
        self.embed2 = ConvTokenEmbedding(c1, c2, k2, s2)
        self.blocks2 = nn.ModuleList([
            ConvTransformerBlock(in_ch=c2, dim=c2, k=kp2, num_heads=H2, mlp_ratio=R2,
                                 attn_drop=attn_drop, proj_drop=proj_drop)
            for _ in range(depth2)
        ])

        # ----------------
        # Stage 3
        # ----------------
        self.embed3 = ConvTokenEmbedding(c2, c3, k3, s3, add_cls_token=True)
        self.blocks3 = nn.ModuleList([
            ConvTransformerBlock(in_ch=c3, dim=c3, k=kp3, num_heads=H3, mlp_ratio=R3,
                                 attn_drop=attn_drop, proj_drop=proj_drop)
            for _ in range(depth3)
        ])

        # mlp head
        self.mlp_head = nn.Linear(c3, num_classes)

    def forward(self, x: torch.Tensor):
        z1 = self.embed1(x)      # [B, N, D]

        for blk in self.blocks1:
            z1 = blk(z1)[0]

        z2 = self.embed2(z1)
        for blk in self.blocks2:
            z2 = blk(z2)[0]

        # ----------------
        # Stage 3
        # ----------------
        z3 = self.embed3(z2)
        cls, z3 = z3[:, :1, :], z3[:, 1:, :]  # split cls token from spatial patch

        for blk in self.blocks3:
            z3, cls = blk(z3, cls)
            # z3: [B, C3, H, W]
            # cls: [B, 1, C3]

        # mlp head
        cls = cls.squeeze(1)
        return self.mlp_head(cls)

model = CvT(img_ch=3,
            attn_drop=0.1, proj_drop=0.1,
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