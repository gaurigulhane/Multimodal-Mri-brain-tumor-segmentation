"""
attention_modules.py
--------------------
Attention mechanisms for the Multimodal MRI Brain Tumor Segmentation project.

Modules:
    2D:
        ChannelAttention   - Squeeze-and-Excitation channel attention
        SpatialAttention   - Spatial attention with avg+max pooled features
        CBAM               - Convolutional Block Attention Module (Channel + Spatial)
        AttentionGate      - Additive attention gate for skip connections

    3D:
        ChannelAttention3D  - SE-style channel attention using 3-D pooling
        SpatialAttention3D  - Spatial attention for 3-D feature maps
        AttentionGate3D     - 3-D attention gate for skip connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 2-D Attention Modules
# ---------------------------------------------------------------------------

class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention (2-D).

    Applies global average pooling AND global max pooling, passes each through
    a shared two-layer MLP, sums the results, and applies a sigmoid gate to
    re-weight the input channels.

    Args:
        in_channels (int): Number of input (and output) feature channels.
        reduction   (int): Reduction ratio for the hidden MLP layer (default 16).
    """

    def __init__(self, in_channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(1, in_channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # Shared MLP – implemented as two 1×1 convolutions (no bias in first).
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            Attended feature map of shape (B, C, H, W).
        """
        avg_out = self.mlp(self.avg_pool(x))   # (B, C, 1, 1)
        max_out = self.mlp(self.max_pool(x))   # (B, C, 1, 1)
        scale = self.sigmoid(avg_out + max_out)  # (B, C, 1, 1)
        return x * scale


class SpatialAttention(nn.Module):
    """Spatial attention with a 7×7 (or custom) convolution.

    Concatenates channel-wise average-pool and max-pool maps, then uses a
    single convolution + sigmoid to produce a spatial gate.

    Args:
        kernel_size (int): Convolution kernel size (must be odd, default 7).
    """

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd for symmetric padding."
        padding = kernel_size // 2
        # 2 input channels: one avg-pool map, one max-pool map
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            Attended feature map of shape (B, C, H, W).
        """
        avg_out = torch.mean(x, dim=1, keepdim=True)   # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)
        pooled = torch.cat([avg_out, max_out], dim=1)   # (B, 2, H, W)
        scale = self.sigmoid(self.conv(pooled))          # (B, 1, H, W)
        return x * scale


class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM).

    Sequentially applies ChannelAttention then SpatialAttention.

    Args:
        in_channels (int): Number of input channels.
        reduction   (int): Reduction ratio for channel attention MLP (default 16).
        kernel_size (int): Kernel size for spatial attention conv (default 7).
    """

    def __init__(self, in_channels: int, reduction: int = 16, kernel_size: int = 7) -> None:
        super().__init__()
        self.channel_att = ChannelAttention(in_channels, reduction=reduction)
        self.spatial_att = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            Attended feature map of shape (B, C, H, W).
        """
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


class AttentionGate(nn.Module):
    """Additive soft attention gate for 2-D U-Net skip connections.

    Combines a gating signal *g* (from the decoder) and the skip-connection
    feature map *x* (from the encoder), produces an attention coefficient α,
    and returns the re-weighted skip features together with the attention map.

    Reference: Oktay et al., "Attention U-Net" (2018).

    Args:
        F_g  (int): Number of channels in the gating signal *g*.
        F_l  (int): Number of channels in the skip-connection feature *x*.
        F_int(int): Number of intermediate channels (hidden dimension).
    """

    def __init__(self, F_g: int, F_l: int, F_int: int) -> None:
        super().__init__()
        # Linear transforms for gating signal and skip features
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor):
        """
        Args:
            g: Gating signal from decoder  (B, F_g, H, W)
            x: Skip-connection features    (B, F_l, H, W)
        Returns:
            attended_x   : Re-weighted skip features (B, F_l, H, W)
            attention_map: Attention coefficients    (B, 1,   H, W)
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # Upsample g1 to match x1 spatial size if needed
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode="bilinear", align_corners=False)
        psi = self.relu(g1 + x1)
        attention_map = self.psi(psi)   # (B, 1, H, W)
        attended_x = x * attention_map  # (B, F_l, H, W)
        return attended_x, attention_map


# ---------------------------------------------------------------------------
# 3-D Attention Modules
# ---------------------------------------------------------------------------

class ChannelAttention3D(nn.Module):
    """SE-style channel attention for 3-D feature maps.

    Uses AdaptiveAvgPool3d + AdaptiveMaxPool3d, then a shared MLP, sums
    both branches, and applies sigmoid re-weighting.

    Args:
        in_channels (int): Number of input channels.
        reduction   (int): Reduction ratio for the hidden MLP layer (default 16).
    """

    def __init__(self, in_channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(1, in_channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        # Shared MLP via 1×1×1 convolutions
        self.mlp = nn.Sequential(
            nn.Conv3d(in_channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, in_channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, D, H, W)
        Returns:
            Attended feature map of shape (B, C, D, H, W).
        """
        avg_out = self.mlp(self.avg_pool(x))    # (B, C, 1, 1, 1)
        max_out = self.mlp(self.max_pool(x))    # (B, C, 1, 1, 1)
        scale = self.sigmoid(avg_out + max_out)  # (B, C, 1, 1, 1)
        return x * scale


class SpatialAttention3D(nn.Module):
    """Spatial attention for 3-D feature maps.

    Concatenates channel-wise average-pool and max-pool volumes, then applies
    a single 3-D convolution + sigmoid to produce a volumetric spatial gate.

    Args:
        kernel_size (int): Convolution kernel size (must be odd, default 7).
    """

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd for symmetric padding."
        padding = kernel_size // 2
        # 2 input channels: avg-pool map + max-pool map
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, D, H, W)
        Returns:
            Attended feature map of shape (B, C, D, H, W).
        """
        avg_out = torch.mean(x, dim=1, keepdim=True)    # (B, 1, D, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, D, H, W)
        pooled = torch.cat([avg_out, max_out], dim=1)   # (B, 2, D, H, W)
        scale = self.sigmoid(self.conv(pooled))          # (B, 1, D, H, W)
        return x * scale


class AttentionGate3D(nn.Module):
    """Additive soft attention gate for 3-D U-Net skip connections.

    Operates identically to AttentionGate but uses 3-D convolutions and
    trilinear interpolation for spatial alignment.

    Args:
        F_g  (int): Number of channels in the gating signal *g*.
        F_l  (int): Number of channels in the skip-connection feature *x*.
        F_int(int): Number of intermediate channels (hidden dimension).
    """

    def __init__(self, F_g: int, F_l: int, F_int: int) -> None:
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor):
        """
        Args:
            g: Gating signal from decoder  (B, F_g, D, H, W)
            x: Skip-connection features    (B, F_l, D, H, W)
        Returns:
            attended_x   : Re-weighted skip features (B, F_l, D, H, W)
            attention_map: Attention coefficients    (B, 1,   D, H, W)
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # Trilinear upsample g1 to match x1 spatial size if needed
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode="trilinear", align_corners=False)
        psi = self.relu(g1 + x1)
        attention_map = self.psi(psi)   # (B, 1, D, H, W)
        attended_x = x * attention_map  # (B, F_l, D, H, W)
        return attended_x, attention_map


# ---------------------------------------------------------------------------
# Quick sanity-check (run as script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running attention module smoke tests ...\n")

    B = 2

    # --- 2-D tests ---
    x2 = torch.randn(B, 32, 64, 64)
    print(f"Input 2D: {tuple(x2.shape)}")

    ca = ChannelAttention(32, reduction=8)
    out = ca(x2)
    assert out.shape == x2.shape, "ChannelAttention shape mismatch"
    print(f"  ChannelAttention   -> {tuple(out.shape)}  ✓")

    sa = SpatialAttention(kernel_size=7)
    out = sa(x2)
    assert out.shape == x2.shape, "SpatialAttention shape mismatch"
    print(f"  SpatialAttention   -> {tuple(out.shape)}  ✓")

    cbam = CBAM(32, reduction=8)
    out = cbam(x2)
    assert out.shape == x2.shape, "CBAM shape mismatch"
    print(f"  CBAM               -> {tuple(out.shape)}  ✓")

    g2 = torch.randn(B, 64, 32, 32)
    x_skip2 = torch.randn(B, 32, 64, 64)
    ag = AttentionGate(F_g=64, F_l=32, F_int=16)
    attended, amap = ag(g2, x_skip2)
    assert attended.shape == x_skip2.shape
    assert amap.shape == (B, 1, 64, 64)
    print(f"  AttentionGate      -> attended {tuple(attended.shape)}, map {tuple(amap.shape)}  ✓")

    # --- 3-D tests ---
    x3 = torch.randn(B, 32, 16, 32, 32)
    print(f"\nInput 3D: {tuple(x3.shape)}")

    ca3 = ChannelAttention3D(32, reduction=8)
    out = ca3(x3)
    assert out.shape == x3.shape, "ChannelAttention3D shape mismatch"
    print(f"  ChannelAttention3D -> {tuple(out.shape)}  ✓")

    sa3 = SpatialAttention3D(kernel_size=7)
    out = sa3(x3)
    assert out.shape == x3.shape, "SpatialAttention3D shape mismatch"
    print(f"  SpatialAttention3D -> {tuple(out.shape)}  ✓")

    g3 = torch.randn(B, 64, 8, 16, 16)
    x_skip3 = torch.randn(B, 32, 16, 32, 32)
    ag3 = AttentionGate3D(F_g=64, F_l=32, F_int=16)
    attended3, amap3 = ag3(g3, x_skip3)
    assert attended3.shape == x_skip3.shape
    assert amap3.shape == (B, 1, 16, 32, 32)
    print(f"  AttentionGate3D    -> attended {tuple(attended3.shape)}, map {tuple(amap3.shape)}  ✓")

    print("\nAll smoke tests passed!")
