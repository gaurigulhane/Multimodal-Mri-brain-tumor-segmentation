"""
residual_blocks.py
------------------
Residual block building blocks for the Multimodal MRI Brain Tumor Segmentation
project.

Blocks:
    ResidualBlock2D       - Standard 2-D residual block (two 3×3 convs + skip)
    ResidualBlock3D       - Standard 3-D residual block (two 3×3×3 convs + skip)
    BottleneckResidual3D  - 3-D bottleneck block  (1×1×1 → 3×3×3 → 1×1×1)
    MultiScaleBlock3D     - Parallel multi-scale branches fused by projection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 2-D Residual Block
# ---------------------------------------------------------------------------

class ResidualBlock2D(nn.Module):
    """Standard 2-D Residual Block.

    Architecture:
        Conv2d(3×3, BN, ReLU) → Conv2d(3×3, BN) → (+ skip) → ReLU

    The skip connection uses a 1×1 convolution (projection shortcut) whenever
    ``in_channels != out_channels`` or ``stride != 1``.

    Args:
        in_channels  (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride       (int): Stride applied to the first convolution and the
                            projection shortcut (default 1).
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Projection shortcut if dimensions change
        self.shortcut: nn.Module
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, H, W)
        Returns:
            (B, out_channels, H', W')  where H'=H//stride, W'=W//stride.
        """
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)
        return out


# ---------------------------------------------------------------------------
# 3-D Residual Block
# ---------------------------------------------------------------------------

class ResidualBlock3D(nn.Module):
    """Standard 3-D Residual Block.

    Architecture:
        Conv3d(3×3×3, BN, ReLU) → Conv3d(3×3×3, BN) → (+ skip) → ReLU

    The skip connection uses a 1×1×1 projection shortcut whenever the channel
    count or spatial dimensions change.

    Args:
        in_channels  (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride       (int): Stride applied to the first conv and shortcut (default 1).
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm3d(out_channels)

        # Projection shortcut
        self.shortcut: nn.Module
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, D, H, W)
        Returns:
            (B, out_channels, D', H', W')
        """
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)
        return out


# ---------------------------------------------------------------------------
# 3-D Bottleneck Residual Block
# ---------------------------------------------------------------------------

class BottleneckResidual3D(nn.Module):
    """3-D Bottleneck Residual Block.

    Architecture:
        Conv3d(1×1×1, BN, ReLU)       [channel compression]
        → Conv3d(3×3×3, BN, ReLU)     [spatial processing]
        → Conv3d(1×1×1, BN)           [channel expansion]
        → (+ skip) → ReLU

    Args:
        in_channels        (int):  Number of input channels.
        out_channels       (int):  Number of output channels.
        bottleneck_channels(int | None): Intermediate channel width.  Defaults
                                         to ``out_channels // 4`` (min 1).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bottleneck_channels: int | None = None,
    ) -> None:
        super().__init__()
        if bottleneck_channels is None:
            bottleneck_channels = max(1, out_channels // 4)

        # 1×1×1 compression
        self.conv1 = nn.Conv3d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(bottleneck_channels)

        # 3×3×3 spatial
        self.conv2 = nn.Conv3d(
            bottleneck_channels, bottleneck_channels,
            kernel_size=3, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm3d(bottleneck_channels)

        # 1×1×1 expansion
        self.conv3 = nn.Conv3d(bottleneck_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # Shortcut projection if channel sizes differ
        self.shortcut: nn.Module
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, D, H, W)
        Returns:
            (B, out_channels, D, H, W)
        """
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out = out + identity
        out = self.relu(out)
        return out


# ---------------------------------------------------------------------------
# 3-D Multi-Scale Block
# ---------------------------------------------------------------------------

class MultiScaleBlock3D(nn.Module):
    """3-D Multi-Scale Residual Block.

    Processes feature maps in parallel with three branches at different effective
    receptive fields, concatenates the branch outputs, and projects back to
    ``out_channels``.

    Branches:
        1. 1×1×1 conv  (point-wise, scale = 1)
        2. 3×3×3 conv  (local context, scale = 3)
        3. 3×3×3 conv with dilation=2  (dilated, effective scale ≈ 5)

    The three branch outputs are concatenated along the channel dimension and
    then projected to ``out_channels`` via a 1×1×1 conv + BN + ReLU.  A
    skip connection (with projection if necessary) is added after the fusion.

    Args:
        in_channels  (int): Number of input channels.
        out_channels (int): Number of output channels.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        # Each branch uses in_channels // 3 intermediate channels (min 1)
        branch_ch = max(1, in_channels // 3)

        # Branch 1: 1×1×1 (point-wise)
        self.branch1 = nn.Sequential(
            nn.Conv3d(in_channels, branch_ch, kernel_size=1, bias=False),
            nn.BatchNorm3d(branch_ch),
            nn.ReLU(inplace=True),
        )

        # Branch 2: 3×3×3 standard conv
        self.branch2 = nn.Sequential(
            nn.Conv3d(in_channels, branch_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(branch_ch),
            nn.ReLU(inplace=True),
        )

        # Branch 3: 3×3×3 dilated conv (dilation=2) → effective 5×5×5 receptive field
        self.branch3 = nn.Sequential(
            nn.Conv3d(in_channels, branch_ch, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm3d(branch_ch),
            nn.ReLU(inplace=True),
        )

        # Fusion projection: concatenated channels → out_channels
        fused_channels = branch_ch * 3
        self.fusion = nn.Sequential(
            nn.Conv3d(fused_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Skip connection
        self.shortcut: nn.Module
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, D, H, W)
        Returns:
            (B, out_channels, D, H, W)
        """
        identity = self.shortcut(x)

        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)

        fused = torch.cat([b1, b2, b3], dim=1)  # (B, 3*branch_ch, D, H, W)
        out = self.fusion(fused)                  # (B, out_channels, D, H, W)

        out = out + identity
        out = self.relu(out)
        return out


# ---------------------------------------------------------------------------
# Quick sanity-check (run as script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running residual block smoke tests ...\n")

    B = 2

    # --- ResidualBlock2D ---
    x2 = torch.randn(B, 32, 64, 64)
    rb2 = ResidualBlock2D(32, 64, stride=2)
    out = rb2(x2)
    assert out.shape == (B, 64, 32, 32), f"Expected (2,64,32,32), got {out.shape}"
    print(f"ResidualBlock2D (32→64, stride=2) : {tuple(x2.shape)} → {tuple(out.shape)}  ✓")

    rb2_same = ResidualBlock2D(64, 64)
    out = rb2_same(out)
    assert out.shape == (B, 64, 32, 32)
    print(f"ResidualBlock2D (64→64, stride=1) : same-dim skip  ✓")

    # --- ResidualBlock3D ---
    x3 = torch.randn(B, 32, 16, 32, 32)
    rb3 = ResidualBlock3D(32, 64, stride=2)
    out = rb3(x3)
    assert out.shape == (B, 64, 8, 16, 16), f"Expected (2,64,8,16,16), got {out.shape}"
    print(f"ResidualBlock3D (32→64, stride=2) : {tuple(x3.shape)} → {tuple(out.shape)}  ✓")

    rb3_same = ResidualBlock3D(64, 64)
    out = rb3_same(out)
    assert out.shape == (B, 64, 8, 16, 16)
    print(f"ResidualBlock3D (64→64, stride=1) : same-dim skip  ✓")

    # --- BottleneckResidual3D ---
    x_btn = torch.randn(B, 256, 8, 8, 8)
    btn = BottleneckResidual3D(256, 512)
    out = btn(x_btn)
    assert out.shape == (B, 512, 8, 8, 8), f"Expected (2,512,8,8,8), got {out.shape}"
    print(f"BottleneckResidual3D (256→512)   : {tuple(x_btn.shape)} → {tuple(out.shape)}  ✓")

    btn_same = BottleneckResidual3D(512, 512, bottleneck_channels=128)
    out = btn_same(out)
    assert out.shape == (B, 512, 8, 8, 8)
    print(f"BottleneckResidual3D (512→512)   : same-dim skip  ✓")

    # --- MultiScaleBlock3D ---
    x_ms = torch.randn(B, 128, 8, 8, 8)
    ms = MultiScaleBlock3D(128, 256)
    out = ms(x_ms)
    assert out.shape == (B, 256, 8, 8, 8), f"Expected (2,256,8,8,8), got {out.shape}"
    print(f"MultiScaleBlock3D   (128→256)    : {tuple(x_ms.shape)} → {tuple(out.shape)}  ✓")

    ms_same = MultiScaleBlock3D(256, 256)
    out = ms_same(out)
    assert out.shape == (B, 256, 8, 8, 8)
    print(f"MultiScaleBlock3D   (256→256)    : same-dim skip  ✓")

    print("\nAll smoke tests passed!")
