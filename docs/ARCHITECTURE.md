# Architectural Specification: Improved Residual Attention U-Net

## 1. High-Level System Overview
This project implements **Multimodal MRI-Based Brain Tumor Segmentation Using an Improved Attention-Based Deep Learning Framework**.

The model processes four co-registered MRI modalities—**T1, T1ce (T1-Gd), T2, and FLAIR**—and outputs voxel-level segmentations for hierarchical tumor subregions:
- **Whole Tumor (WT)**: Extent of entire disease including peritumoral edema / SNFH.
- **Tumor Core (TC)**: Gross tumor mass excluding non-infiltrated edema.
- **Enhancing Tumor (ET)**: Actively proliferating, hypervascular tumor rim.

```mermaid
flowchart TD
    subgraph Inputs["Multimodal MRI Inputs (B, 4, D, H, W)"]
        T1["T1 (Anatomy)"]
        T1ce["T1ce (Enhancement)"]
        T2["T2 (Fluid/Edema)"]
        FLAIR["FLAIR (Edema/Hyperintensity)"]
    end

    Inputs --> Stacking["4-Channel Stacking"]
    Stacking --> E1["Encoder Level 1 (32 ch)"]
    E1 --> E2["Encoder Level 2 (64 ch)"]
    E2 --> E3["Encoder Level 3 (128 ch)"]
    E3 --> E4["Encoder Level 4 (256 ch)"]

    E4 --> Bottleneck["Multi-Scale Bottleneck (512 ch)"]

    Bottleneck --> D4["Decoder Level 4 (256 ch)"]
    D4 --> D3["Decoder Level 3 (128 ch)"]
    D3 --> D2["Decoder Level 2 (64 ch)"]
    D2 --> D1["Decoder Level 1 (32 ch)"]

    E4 -.-> AG4["Attention Gate 4"] .-> D4
    E3 -.-> AG3["Attention Gate 3"] .-> D3
    E2 -.-> AG2["Attention Gate 2"] .-> D2
    E1 -.-> AG1["Attention Gate 1"] .-> D1

    D1 --> Head["1x1x1 Conv + Sigmoid Head"]
    Head --> Out["Multi-Label Output: [WT, TC, ET] (B, 3, D, H, W)"]
```

---

## 2. Encoder: 3D Residual Block + CBAM Attention
Each level of the encoder is composed of:
1. **3D Residual Block**: Two consecutive $3 \times 3 \times 3$ convolutions with Batch Normalization and ReLU activation, plus a shortcut identity or $1 \times 1 \times 1$ projection:
   $$y = \text{ReLU}\left(\mathcal{F}(x, \{W_i\}) + W_s x\right)$$
2. **Channel Attention (SE-Net)**:
   $$\mathbf{M}_c(\mathbf{F}) = \sigma\left(\mathbf{W}_1 \text{ReLU}\left(\mathbf{W}_0(\mathbf{F}_{\text{avg}}^c)\right) + \mathbf{W}_1 \text{ReLU}\left(\mathbf{W}_0(\mathbf{F}_{\text{max}}^c)\right)\right)$$
3. **Spatial Attention**:
   $$\mathbf{M}_s(\mathbf{F}) = \sigma\left(f^{7\times 7\times 7}\left(\left[\mathbf{F}_{\text{avg}}^s; \mathbf{F}_{\text{max}}^s\right]\right)\right)$$

```mermaid
flowchart LR
    In["Feature Map X"] --> Res["3D Residual Block"]
    Res --> CA["Channel Attention (SE-Net)"]
    CA --> SA["3D Spatial Attention (7x7x7)"]
    SA --> Out["Attended Features X'"]
```

---

## 3. Attention Gates (AG) on Skip Connections
Additive attention gates gate skip features $x^l$ before concatenation with upsampled decoder features $g$:
$$\alpha = \sigma\left(\psi^T\left(\text{ReLU}\left(W_g g + W_x x^l + b_g + b_x\right)\right) + b_\psi\right)$$
$$\hat{x}^l = \alpha \odot x^l$$

---

## 4. Bottleneck: Multi-Scale Dilated Receptive Field
At the bottleneck (512 features), parallel convolutions capture local and multi-scale context:
- Branch 1: $3 \times 3 \times 3$ standard convolution ($d=1$)
- Branch 2: $3 \times 3 \times 3$ dilated convolution ($d=2$)
- Branch 3: $1 \times 1 \times 1$ linear feature projection
- Concat + $1 \times 1 \times 1$ fusion

---

## 5. Dataset Taxonomy & Clinical Ground Truth
| Dataset | Modalities | Raw Labels | Output Regions |
| :--- | :--- | :--- | :--- |
| **BraTS 2021** (Glioma) | T1, T1ce, T2, FLAIR | 0: Background<br>1: NCR (Necrotic Core)<br>2: ED (Edema)<br>3: ET (Enhancing Tumor) | **WT**: 1 + 2 + 3<br>**TC**: 1 + 3<br>**ET**: 3 |
| **BraTS 2023-MET** (Metastasis) | T1, T1ce, T2, FLAIR | 0: Background<br>1: NETC (Non-Enhancing)<br>2: SNFH (Hyperintensity)<br>3: ET (Enhancing Focus) | **WT**: 1 + 2 + 3<br>**TC**: 1 + 3<br>**ET**: 3 |
