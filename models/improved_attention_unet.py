"""
improved_attention_unet.py
--------------------------
Improved Residual Attention U-Net (3-D) for Multimodal MRI Brain Tumor
Segmentation.

Classes:
    ImprovedResAttentionUNet3D  - Main proposed model combining residual blocks,
                                   CBAM-style 3-D attention, and soft attention
                                   gates on skip connections.

Functions:
    get_improved_model() -> ImprovedResAttentionUNet3D
    get_baseline_model() -> UNet3D   (re-exported for convenience)
    get_model(model_type)            -> either model based on string selector

Architecture summary
────────────────────
Input  : (B, 4, D, H, W)  — T1, T1ce, T2, FLAIR

Encoder (4 levels, features = [32, 64, 128, 256])
    Each level:
        ResidualBlock3D  → ChannelAttention3D → SpatialAttention3D
    Followed by MaxPool3d(2) downsampling between levels.

Bottleneck
    BottleneckResidual3D(256 → 512) → MultiScaleBlock3D(512 → 512)

Decoder (4 levels)
    Each level:
        ConvTranspose3d(2×) upsample
        → AttentionGate3D applied to corresponding encoder skip
        → Concatenate gated skip + upsampled feature
        → ResidualBlock3D
    Output channels: 512→256→128→64→32

Segmentation head
    Conv3d(1×1×1): 32 → 3  (WT / TC / ET, multi-label sigmoid)

Forward output
    {
        'logits'        : Tensor  (B, 3, D, H, W),
        'attention_maps': list of 4 Tensors, each (B, 1, Dk, Hk, Wk)
    }
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Local imports from the same package
from .attention_modules import (
    AttentionGate3D,
    ChannelAttention3D,
    SpatialAttention3D,
)
from .residual_blocks import (
    BottleneckResidual3D,
    MultiScaleBlock3D,
    ResidualBlock3D,
)
from .baseline_model import UNet3D


# ---------------------------------------------------------------------------
# Encoder block: ResidualBlock3D + ChannelAttention3D + SpatialAttention3D
# ---------------------------------------------------------------------------

class _EncoderBlock(nn.Module):
    """Single encoder level combining residual convolutions with 3-D CBAM attention.

    Args:
        in_channels  (int): Input channel count.
        out_channels (int): Output channel count.
        reduction    (int): Channel-attention reduction ratio (default 16).
    """

    def __init__(self, in_channels: int, out_channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.res_block = ResidualBlock3D(in_channels, out_channels)
        self.channel_att = ChannelAttention3D(out_channels, reduction=reduction)
        self.spatial_att = SpatialAttention3D(kernel_size=7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, D, H, W)
        Returns:
            (B, out_channels, D, H, W)
        """
        x = self.res_block(x)
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


# ---------------------------------------------------------------------------
# Decoder block: AttentionGate3D gated skip + ResidualBlock3D
# ---------------------------------------------------------------------------

class _DecoderBlock(nn.Module):
    """Single decoder level with attention-gated skip connection.

    Steps:
        1. Up-sample with ConvTranspose3d (stride=2).
        2. Apply AttentionGate3D to the encoder skip using the upsampled
           feature as the gating signal.
        3. Concatenate gated skip + upsampled feature.
        4. Apply ResidualBlock3D to the concatenated tensor.

    Args:
        in_channels   (int): Channels coming from the previous decoder level
                              (= upconv input channels).
        skip_channels (int): Channels from the corresponding encoder skip.
        out_channels  (int): Output channels after the decoder ResidualBlock3D.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        # Transpose conv: halves spatial dims, maps in_channels → skip_channels
        self.upconv = nn.ConvTranspose3d(
            in_channels, skip_channels, kernel_size=2, stride=2
        )
        # Attention gate: gating signal is upsampled (skip_channels),
        # skip features are encoder (skip_channels)
        att_int = max(1, skip_channels // 2)
        self.attention_gate = AttentionGate3D(
            F_g=skip_channels, F_l=skip_channels, F_int=att_int
        )
        # After concat: 2 * skip_channels → out_channels
        self.res_block = ResidualBlock3D(skip_channels * 2, out_channels)

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x   : Decoder feature map from previous level (B, in_channels, D, H, W).
            skip: Encoder skip connection            (B, skip_channels, D', H', W').
        Returns:
            out         : (B, out_channels, D', H', W')
            attention_map: (B, 1, D', H', W')
        """
        x = self.upconv(x)

        # Handle potential odd-dimension mismatch
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)

        # Gated skip: use upsampled x as gating signal, encoder skip as value
        gated_skip, attention_map = self.attention_gate(g=x, x=skip)

        # Concatenate and apply residual block
        x = torch.cat([gated_skip, x], dim=1)
        out = self.res_block(x)
        return out, attention_map


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------

class ImprovedResAttentionUNet3D(nn.Module):
    """Improved 3-D Residual Attention U-Net for Brain Tumour Segmentation.

    Combines:
        • ResidualBlock3D encoder stages with CBAM-style 3-D attention.
        • BottleneckResidual3D + MultiScaleBlock3D bottleneck for rich
          multi-scale representations.
        • AttentionGate3D gated skip connections in the decoder.
        • Multi-label sigmoid segmentation for WT / TC / ET tumour regions.

    Args:
        in_channels         (int):       Input MRI modalities (default 4).
        out_channels        (int):       Segmentation classes (default 3).
        features            (list[int]): Encoder feature widths
                                          (default [32, 64, 128, 256]).
        bottleneck_features (int):       Bottleneck feature width (default 512).
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        features: list[int] | None = None,
        bottleneck_features: int = 512,
    ) -> None:
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features
        self.bottleneck_features = bottleneck_features

        # ---- Encoder --------------------------------------------------------
        # Reduction capped at a safe value relative to the first (smallest)
        # feature width to avoid hidden=0.
        safe_reduction = min(16, features[0])

        self.encoder_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()

        prev_ch = in_channels
        for feat in features:
            self.encoder_blocks.append(_EncoderBlock(prev_ch, feat, reduction=safe_reduction))
            self.pools.append(nn.MaxPool3d(kernel_size=2, stride=2))
            prev_ch = feat

        # ---- Bottleneck -----------------------------------------------------
        # BottleneckResidual3D: features[-1] → bottleneck_features
        self.bottleneck_res = BottleneckResidual3D(features[-1], bottleneck_features)
        # MultiScaleBlock3D: bottleneck_features → bottleneck_features
        self.bottleneck_ms = MultiScaleBlock3D(bottleneck_features, bottleneck_features)

        # ---- Decoder --------------------------------------------------------
        # decoder_block[i] up-samples from features[-1-i] level.
        # Channels flow: 512 → 256 → 128 → 64 → 32
        self.decoder_blocks = nn.ModuleList()

        dec_in = bottleneck_features   # 512
        for feat in reversed(features):          # 256, 128, 64, 32
            self.decoder_blocks.append(
                _DecoderBlock(
                    in_channels=dec_in,
                    skip_channels=feat,
                    out_channels=feat,
                )
            )
            dec_in = feat

        # ---- Segmentation head ----------------------------------------------
        self.seg_head = nn.Conv3d(features[0], out_channels, kernel_size=1)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | list]:
        """
        Args:
            x: (B, 4, D, H, W) — multimodal MRI input.
        Returns:
            dict with keys:
                'logits'        : Tensor (B, 3, D, H, W)  raw logits.
                'attention_maps': list of 4 Tensors, one per decoder level,
                                  each of shape (B, 1, Dk, Hk, Wk).
        """
        # ---- Encoder path ---------------------------------------------------
        skip_connections: list[torch.Tensor] = []

        for enc_block, pool in zip(self.encoder_blocks, self.pools):
            x = enc_block(x)
            skip_connections.append(x)
            x = pool(x)

        # ---- Bottleneck -----------------------------------------------------
        x = self.bottleneck_res(x)
        x = self.bottleneck_ms(x)

        # ---- Decoder path ---------------------------------------------------
        skip_connections_rev = skip_connections[::-1]  # deepest skip first
        attention_maps: list[torch.Tensor] = []

        for dec_block, skip in zip(self.decoder_blocks, skip_connections_rev):
            x, amap = dec_block(x, skip)
            attention_maps.append(amap)

        # ---- Segmentation head ----------------------------------------------
        logits = self.seg_head(x)  # (B, 3, D, H, W)

        return {
            "logits": logits,
            "attention_maps": attention_maps,
        }

    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        """Returns total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def get_improved_model(
    in_channels: int = 4,
    out_channels: int = 3,
    features: list[int] | None = None,
    bottleneck_features: int = 512,
) -> ImprovedResAttentionUNet3D:
    """Instantiate and return the ImprovedResAttentionUNet3D.

    Prints parameter count to stdout.

    Args:
        in_channels         (int): MRI modalities (default 4).
        out_channels        (int): Segmentation classes (default 3).
        features            (list[int]): Encoder widths (default [32,64,128,256]).
        bottleneck_features (int): Bottleneck width (default 512).

    Returns:
        ImprovedResAttentionUNet3D: Model in training mode.
    """
    if features is None:
        features = [32, 64, 128, 256]

    model = ImprovedResAttentionUNet3D(
        in_channels=in_channels,
        out_channels=out_channels,
        features=features,
        bottleneck_features=bottleneck_features,
    )
    n_params = model.count_parameters()
    print(
        f"[ImprovedResAttentionUNet3D] Model initialised.\n"
        f"  in_channels         : {in_channels}\n"
        f"  out_channels        : {out_channels}\n"
        f"  features            : {features}\n"
        f"  bottleneck_features : {bottleneck_features}\n"
        f"  Trainable parameters: {n_params:,}"
    )
    return model


def get_model(model_type: str = "improved") -> nn.Module:
    """Return either the improved or baseline model by name.

    Args:
        model_type (str): ``'improved'`` (default) or ``'baseline'``.

    Returns:
        nn.Module: The requested model in training mode.

    Raises:
        ValueError: If ``model_type`` is not recognised.
    """
    model_type = model_type.lower().strip()
    if model_type == "improved":
        return get_improved_model()
    elif model_type in ("baseline", "unet3d"):
        from .baseline_model import get_baseline_model as _get_baseline
        return _get_baseline()
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            "Choose from: 'improved', 'baseline' / 'unet3d'."
        )


# ---------------------------------------------------------------------------
# Quick sanity-check (run as script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os

    # Allow running directly without package install by adjusting sys.path
    _parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _parent not in sys.path:
        sys.path.insert(0, _parent)

    # Re-import with explicit package paths for standalone execution
    from models.attention_modules import (  # type: ignore[no-redef]
        AttentionGate3D,
        ChannelAttention3D,
        SpatialAttention3D,
    )
    from models.residual_blocks import (  # type: ignore[no-redef]
        BottleneckResidual3D,
        MultiScaleBlock3D,
        ResidualBlock3D,
    )
    from models.baseline_model import UNet3D  # type: ignore[no-redef]

    print("=" * 60)
    print("Improved Residual Attention U-Net 3D — smoke test")
    print("=" * 60)

    model = ImprovedResAttentionUNet3D(
        in_channels=4,
        out_channels=3,
        features=[32, 64, 128, 256],
        bottleneck_features=512,
    )
    n_params = model.count_parameters()
    print(f"Trainable parameters : {n_params:,}\n")

    model.eval()
    B, D, H, W = 1, 64, 128, 128
    dummy = torch.randn(B, 4, D, H, W)
    print(f"Input shape : {tuple(dummy.shape)}")

    with torch.no_grad():
        output = model(dummy)

    logits = output["logits"]
    attention_maps = output["attention_maps"]

    print(f"Output logits shape   : {tuple(logits.shape)}")
    assert logits.shape == (B, 3, D, H, W), (
        f"Unexpected logits shape: {tuple(logits.shape)}"
    )
    print(f"Number of attention maps : {len(attention_maps)}")
    for i, amap in enumerate(attention_maps):
        print(f"  attention_maps[{i}] : {tuple(amap.shape)}")

    probs = torch.sigmoid(logits)
    assert probs.min() >= 0.0 and probs.max() <= 1.0
    print("\nAll assertions passed  ✓")

    # ---- Compare parameter counts ----------------------------------------
    print("\n" + "=" * 60)
    print("Parameter count comparison")
    print("=" * 60)
    baseline = UNet3D(in_channels=4, out_channels=3, features=[32, 64, 128, 256])
    b_params = sum(p.numel() for p in baseline.parameters() if p.requires_grad)
    print(f"  UNet3D (baseline)             : {b_params:>12,}")
    print(f"  ImprovedResAttentionUNet3D    : {n_params:>12,}")
    print(f"  Additional params             : {n_params - b_params:>+12,}")
