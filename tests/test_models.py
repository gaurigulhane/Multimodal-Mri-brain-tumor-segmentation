"""
Unit and integration tests for the Multimodal MRI Brain Tumor Segmentation models,
losses, attention modules, and metrics.
"""

import pytest
import torch
import numpy as np

from models.baseline_model import UNet3D, get_baseline_model
from models.improved_attention_unet import ImprovedResAttentionUNet3D, get_improved_model, get_model
from models.attention_modules import ChannelAttention3D, SpatialAttention3D, AttentionGate3D
from models.residual_blocks import ResidualBlock3D, BottleneckResidual3D, MultiScaleBlock3D
from training.losses import get_loss_function, DiceLoss, FocalLoss, DiceFocalLoss
from evaluation.metrics import compute_dice, compute_iou, compute_precision, compute_recall, SegmentationMetrics
from datasets.label_mapping import get_label_mapper, map_labels_to_regions, BRATS2021_CONFIG, BRATS2023MET_CONFIG
from inference.predict import TumorPredictor


def test_attention_modules():
    B, C, D, H, W = 1, 32, 16, 16, 16
    x = torch.randn(B, C, D, H, W)
    
    # Channel Attention
    ca = ChannelAttention3D(in_channels=C, reduction=8)
    out_ca = ca(x)
    assert out_ca.shape == x.shape, f"ChannelAttention output shape mismatch: {out_ca.shape}"
    
    # Spatial Attention
    sa = SpatialAttention3D(kernel_size=7)
    out_sa = sa(x)
    assert out_sa.shape == x.shape, f"SpatialAttention output shape mismatch: {out_sa.shape}"
    
    # Attention Gate
    g = torch.randn(B, 64, 8, 8, 8)
    x_skip = torch.randn(B, 32, 16, 16, 16)
    ag = AttentionGate3D(F_g=64, F_l=32, F_int=16)
    out_ag, att_map = ag(g, x_skip)
    assert out_ag.shape == x_skip.shape, f"AttentionGate output shape mismatch: {out_ag.shape}"
    assert att_map.shape == (B, 1, 16, 16, 16), f"AttentionMap shape mismatch: {att_map.shape}"


def test_residual_blocks():
    B, D, H, W = 1, 16, 16, 16
    x = torch.randn(B, 16, D, H, W)
    
    res = ResidualBlock3D(16, 32)
    out = res(x)
    assert out.shape == (B, 32, D, H, W)
    
    bottle = BottleneckResidual3D(32, 64)
    out_b = bottle(out)
    assert out_b.shape == (B, 64, D, H, W)
    
    ms = MultiScaleBlock3D(64, 64)
    out_ms = ms(out_b)
    assert out_ms.shape == (B, 64, D, H, W)


def test_baseline_model():
    model = UNet3D(in_channels=4, out_channels=3, features=[16, 32, 64, 128])
    x = torch.randn(1, 4, 16, 32, 32)
    out = model(x)
    assert out.shape == (1, 3, 16, 32, 32)


def test_improved_attention_unet():
    model = ImprovedResAttentionUNet3D(
        in_channels=4,
        out_channels=3,
        features=[16, 32, 64, 128],
        bottleneck_features=256
    )
    x = torch.randn(1, 4, 16, 32, 32)
    out = model(x)
    assert isinstance(out, dict)
    assert "logits" in out
    assert "attention_maps" in out
    assert out["logits"].shape == (1, 3, 16, 32, 32)
    assert len(out["attention_maps"]) == 4


def test_loss_functions():
    pred = torch.randn(1, 3, 16, 16, 16)
    target = (torch.rand(1, 3, 16, 16, 16) > 0.7).float()
    
    for name in ["dice", "focal", "dice_focal", "dice_ce", "tversky"]:
        loss_fn = get_loss_function(name)
        if name in ["dice", "tversky"]:
            loss_val = loss_fn(torch.sigmoid(pred), target)
        else:
            loss_val = loss_fn(pred, target)
        assert torch.isfinite(loss_val)
        assert loss_val.item() >= 0.0
        
        # Test dict input compatibility
        dict_input = {"logits": pred}
        loss_dict = loss_fn(dict_input if name not in ["dice", "tversky"] else torch.sigmoid(pred), target)
        assert torch.isfinite(loss_dict)


def test_metrics_computation():
    pred_mask = np.zeros((32, 32, 32), dtype=np.uint8)
    gt_mask = np.zeros((32, 32, 32), dtype=np.uint8)
    
    pred_mask[10:20, 10:20, 10:20] = 1
    gt_mask[12:22, 12:22, 12:22] = 1
    
    dice = compute_dice(pred_mask, gt_mask)
    iou = compute_iou(pred_mask, gt_mask)
    prec = compute_precision(pred_mask, gt_mask)
    rec = compute_recall(pred_mask, gt_mask)
    
    assert 0.0 < dice < 1.0
    assert 0.0 < iou < 1.0
    assert 0.0 < prec < 1.0
    assert 0.0 < rec < 1.0


def test_label_mapping():
    assert BRATS2021_CONFIG.dataset_name == "BraTS2021"
    assert "WT" in BRATS2021_CONFIG.region_definitions
    assert "TC" in BRATS2021_CONFIG.region_definitions
    assert "ET" in BRATS2021_CONFIG.region_definitions
    
    assert BRATS2023MET_CONFIG.dataset_name == "BraTS2023-MET"
    
    # Test mapping
    mapper_2021 = get_label_mapper("brats2021")
    raw = np.array([0, 1, 2, 3])
    out = mapper_2021(raw)
    assert out.shape == (3, 4)
    # Channel 0 (WT): [0, 1, 1, 1]
    assert np.array_equal(out[0], [0, 1, 1, 1])
    # Channel 1 (TC): [0, 1, 0, 1]
    assert np.array_equal(out[1], [0, 1, 0, 1])
    # Channel 2 (ET): [0, 0, 0, 1]
    assert np.array_equal(out[2], [0, 0, 0, 1])


def test_predictor_from_arrays():
    predictor = TumorPredictor(checkpoint_path=None, model_type="improved")
    
    # 3D synthetic MRI modalities
    shape = (16, 32, 32)
    t1 = np.random.randn(*shape).astype(np.float32)
    t1ce = np.random.randn(*shape).astype(np.float32)
    t2 = np.random.randn(*shape).astype(np.float32)
    flair = np.random.randn(*shape).astype(np.float32)
    
    result = predictor.predict_from_arrays(t1, t1ce, t2, flair)
    assert "prediction" in result
    assert "probability" in result
    assert "regions" in result
    assert result["prediction"].shape == (3, *shape)
    assert "WT" in result["regions"]
    assert "TC" in result["regions"]
    assert "ET" in result["regions"]
