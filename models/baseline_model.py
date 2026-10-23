"""
baseline_model.py
-----------------
Clean 3-D U-Net baseline for Multimodal MRI Brain Tumor Segmentation.

Classes:
    DoubleConv3D  - Two sequential 3×3×3 Conv + BN + ReLU layers.
    UNet3D        - Full 3-D U-Net with 4-level encoder / decoder.

Functions:
    get_baseline_model() -> UNet3D   Returns a freshly initialised UNet3D and
                                      prints its total trainable parameter count.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helper: Double Convolution Block
# ---------------------------------------------------------------------------

class DoubleConv3D(nn.Module):
    """Two consecutive 3×3×3 Conv → BN → ReLU layers.

    Optionally applies a 1×1×1 projection on the input as a skip connection
    when ``residual=True`` and channel counts differ.

    Args:
        in_channels  (int):  Input channel count.
        out_channels (int):  Output channel count.
        residual     (bool): If True, adds a skip connection (default False).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        residual: bool = False,
    ) -> None:
        super().__init__()
        self.residual = residual

        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)

        # Projection shortcut for optional residual connection
        if residual:
            if in_channels != out_channels:
                self.shortcut: nn.Module = nn.Sequential(
                    nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm3d(out_channels),
                )
            else:
                self.shortcut = nn.Identity()
        else:
            self.shortcut = None  # type: ignore[assignment]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, D, H, W)
        Returns:
            (B, out_channels, D, H, W)
        """
        out = self.double_conv(x)
        if self.residual and self.shortcut is not None:
            out = out + self.shortcut(x)
        return self.relu(out)


# ---------------------------------------------------------------------------
# Baseline 3-D U-Net
# ---------------------------------------------------------------------------

class UNet3D(nn.Module):
    """3-D U-Net Baseline for Brain Tumor Segmentation.

    Architecture overview
    ─────────────────────
    Encoder (4 levels)
        Each level consists of a ``DoubleConv3D`` block followed by
        ``MaxPool3d(2)`` downsampling (except after the last encoder level).
        Feature map widths: [32, 64, 128, 256].

    Bottleneck
        A single ``DoubleConv3D`` at 512 feature channels.

    Decoder (4 levels)
        Each level up-samples with ``ConvTranspose3d(stride=2)``, concatenates
        the corresponding encoder skip connection, then applies ``DoubleConv3D``.
        Feature map widths (after skip concat): 512→256, 256→128, 128→64, 64→32.

    Segmentation head
        1×1×1 ``Conv3d`` mapping 32 → ``out_channels`` (3) for WT / TC / ET
        multi-label segmentation.

    Args:
        in_channels  (int):       Number of input MRI modalities (default 4).
        out_channels (int):       Number of segmentation classes (default 3).
        features     (list[int]): Feature widths at each encoder level
                                  (default [32, 64, 128, 256]).
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        features: list[int] | None = None,
    ) -> None:
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features

        # ---- Encoder --------------------------------------------------------
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()

        prev_ch = in_channels
        for feat in features:
            self.encoders.append(DoubleConv3D(prev_ch, feat))
            self.pools.append(nn.MaxPool3d(kernel_size=2, stride=2))
            prev_ch = feat

        # ---- Bottleneck -----------------------------------------------------
        bottleneck_feat = features[-1] * 2  # 512
        self.bottleneck = DoubleConv3D(features[-1], bottleneck_feat)

        # ---- Decoder --------------------------------------------------------
        # Levels from deepest (nearest bottleneck) to shallowest
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        dec_in = bottleneck_feat  # starts at 512
        for feat in reversed(features):
            # Transpose conv halves spatial dims, reduces channels to feat
            self.upconvs.append(
                nn.ConvTranspose3d(dec_in, feat, kernel_size=2, stride=2)
            )
            # After concatenation with skip: feat*2 channels
            self.decoders.append(DoubleConv3D(feat * 2, feat))
            dec_in = feat

        # ---- Segmentation head ----------------------------------------------
        self.seg_head = nn.Conv3d(features[0], out_channels, kernel_size=1)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, 4, D, H, W).
        Returns:
            logits: (B, 3, D, H, W) — raw logits (apply sigmoid for inference).
        """
        # ---- Encoder path ---------------------------------------------------
        skip_connections: list[torch.Tensor] = []

        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            skip_connections.append(x)
            x = pool(x)

        # ---- Bottleneck -----------------------------------------------------
        x = self.bottleneck(x)

        # ---- Decoder path ---------------------------------------------------
        skip_connections = skip_connections[::-1]  # reverse: deepest first

        for upconv, decoder, skip in zip(self.upconvs, self.decoders, skip_connections):
            x = upconv(x)
            # Handle potential size mismatch due to odd spatial dimensions
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = decoder(x)

        # ---- Segmentation head ----------------------------------------------
        logits = self.seg_head(x)   # (B, 3, D, H, W)
        return logits

    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        """Returns total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def get_baseline_model(
    in_channels: int = 4,
    out_channels: int = 3,
    features: list[int] | None = None,
) -> UNet3D:
    """Instantiate and return a UNet3D baseline model.

    Prints the total trainable parameter count to stdout.

    Args:
        in_channels  (int):       MRI input modalities (default 4).
        out_channels (int):       Segmentation output classes (default 3).
        features     (list[int]): Encoder feature widths (default [32,64,128,256]).

    Returns:
        UNet3D: Initialised model in training mode.
    """
    if features is None:
        features = [32, 64, 128, 256]

    model = UNet3D(in_channels=in_channels, out_channels=out_channels, features=features)
    n_params = model.count_parameters()
    print(
        f"[UNet3D] Baseline model initialised.\n"
        f"  in_channels  : {in_channels}\n"
        f"  out_channels : {out_channels}\n"
        f"  features     : {features}\n"
        f"  Trainable parameters: {n_params:,}"
    )
    return model


# ---------------------------------------------------------------------------
# Quick sanity-check (run as script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = get_baseline_model()
    model.eval()

    B, D, H, W = 1, 64, 128, 128
    dummy = torch.randn(B, 4, D, H, W)
    print(f"\nInput  : {tuple(dummy.shape)}")

    with torch.no_grad():
        out = model(dummy)
    print(f"Output : {tuple(out.shape)}")

    assert out.shape == (B, 3, D, H, W), (
        f"Expected output shape (1, 3, 64, 128, 128), got {tuple(out.shape)}"
    )
    print("Shape assertion passed  ✓")

    # Verify sigmoid output is in [0, 1]
    probs = torch.sigmoid(out)
    assert probs.min() >= 0.0 and probs.max() <= 1.0
    print("Sigmoid output range [0,1] verified  ✓")
