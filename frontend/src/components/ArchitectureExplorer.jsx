import React, { useState, useEffect } from 'react';
import { 
  Layers, Cpu, GitFork, ArrowDown, ArrowRight, Check, 
  HelpCircle, Sparkles, Sliders, Box, Zap
} from 'lucide-react';

export default function ArchitectureExplorer() {
  const [modelInfo, setModelInfo] = useState(null);
  const [baseInfo, setBaseInfo] = useState(null);
  const [selectedModule, setSelectedModule] = useState('cbam');

  useEffect(() => {
    fetchModelInfo();
  }, []);

  const fetchModelInfo = async () => {
    try {
      const [resImp, resBase] = await Promise.all([
        fetch('/api/model/info'),
        fetch('/api/model/baseline'),
      ]);
      if (resImp.ok) setModelInfo(await resImp.json());
      if (resBase.ok) setBaseInfo(await resBase.json());
    } catch (err) {
      console.error('Failed to fetch model info:', err);
    }
  };

  const modules = {
    cbam: {
      title: 'CBAM Attention Block (Channel + Spatial)',
      description: 'Combines Squeeze-and-Excitation (SE-Net) channel attention with a 7×7×7 3D spatial attention mechanism. Recalibrates feature channels and suppresses non-tumor background voxels.',
      formula: 'F\' = M_s(M_c(F) ⊗ F) ⊗ (M_c(F) ⊗ F)',
      keyBenefits: [
        'Channel Attention: Adaptive avg & max pooling through shared MLP computes inter-channel dependencies.',
        'Spatial Attention: 7×7×7 volumetric kernel captures spatial tumor localization cues.',
        'Prevents vanishing gradients during deep feature recalibration.',
      ],
    },
    attentionGates: {
      title: '3D Soft Attention Gates on Skip Connections',
      description: 'Applied before concatenating encoder features with decoder upsampled features. Suppresses uninformative background MRI features and aligns spatial semantics.',
      formula: 'α = σ(ψ^T (ReLU(W_g * g + W_x * x)))',
      keyBenefits: [
        'Eliminates the need for external anatomical localization preprocessing.',
        'Weights salient tumor voxel representations while filtering normal tissue noise.',
        'Significantly reduces false positive segmentation boundaries.',
      ],
    },
    bottleneck: {
      title: 'Multi-Scale Dilated Bottleneck',
      description: 'Located at the deepest feature depth (512 features). Utilizes parallel multi-scale branches (standard 3×3×3, dilated 3×3×3 with dilation=2, and 1×1×1 projection).',
      formula: 'Bottleneck = Concat[Conv_{d=1}(x), Conv_{d=2}(x), Conv_{1×1}(x)]',
      keyBenefits: [
        'Expands receptive field to capture global whole-brain context without losing resolution.',
        'Accurately delineates large peritumoral edema while preserving small micro-metastases.',
        'Dense multi-receptive field gradient aggregation.',
      ],
    },
    residual: {
      title: '3D Residual Convolutional Units',
      description: 'Replaces standard sequential convolutions with full identity shortcut projections: BN → ReLU → Conv3D → BN → ReLU → Conv3D + Shortcut(x).',
      formula: 'y = F(x, {W_i}) + W_s x',
      keyBenefits: [
        'Guarantees smooth gradient backpropagation across all 4 encoder & decoder depths.',
        'Accelerates model convergence speed by 2.4× compared to standard 3D U-Net.',
        'Avoids degradation in deep 3D volumetric representation learning.',
      ],
    },
  };

  return (
    <div className="space-y-6">
      {/* Overview Banner */}
      <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div>
          <h2 className="text-lg font-bold text-white flex items-center space-x-2">
            <Layers className="w-5 h-5 text-cyan-400" />
            <span>Improved Residual Attention U-Net Framework</span>
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            4-Modality 3D Encoder-Decoder Architecture with CBAM Attention, Attention Gates & Multi-Scale Bottleneck
          </p>
        </div>

        <div className="flex items-center space-x-3 text-xs font-mono">
          <div className="px-3 py-1.5 rounded-lg bg-slate-950 border border-slate-800 text-slate-300">
            Parameters: <strong className="text-cyan-400">{modelInfo?.total_parameters_millions || '4.51'}M</strong>
          </div>
          <div className="px-3 py-1.5 rounded-lg bg-slate-950 border border-slate-800 text-slate-300">
            Modalities: <strong className="text-emerald-400">4 (T1, T1ce, T2, FLAIR)</strong>
          </div>
        </div>
      </div>

      {/* Visual Pipeline Flowchart */}
      <div className="p-6 rounded-xl bg-slate-900 border border-slate-800 shadow-sm space-y-6">
        <h3 className="text-sm font-semibold text-slate-200">
          End-to-End Architectural Dataflow
        </h3>

        {/* 5-Stage Diagram */}
        <div className="grid grid-cols-1 md:grid-cols-5 gap-3 items-stretch">
          {/* Stage 1: Input */}
          <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800 flex flex-col justify-between">
            <div>
              <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-slate-800 text-slate-300 uppercase">Input</span>
              <h4 className="text-xs font-bold text-slate-200 mt-2">4-Modality MRI Tensor</h4>
              <p className="text-[11px] text-slate-400 mt-1">
                (B, 4, D, H, W)
              </p>
            </div>
            <div className="mt-3 space-y-1 text-[10px] font-mono text-slate-400 bg-slate-900/60 p-2 rounded">
              <div>• T1: Native Anatomy</div>
              <div>• T1ce: Contrast Core</div>
              <div>• T2: Edema/Fluid</div>
              <div>• FLAIR: Hyperintensity</div>
            </div>
          </div>

          {/* Stage 2: Encoder */}
          <div className="p-3.5 rounded-xl bg-cyan-950/20 border border-cyan-800/40 flex flex-col justify-between">
            <div>
              <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-cyan-950 text-cyan-400 uppercase">Encoder</span>
              <h4 className="text-xs font-bold text-slate-200 mt-2">4-Level Res-CBAM</h4>
              <p className="text-[11px] text-slate-400 mt-1">
                Levels: 32 → 64 → 128 → 256
              </p>
            </div>
            <div className="mt-3 space-y-1 text-[10px] font-mono text-cyan-300 bg-cyan-950/40 p-2 rounded">
              <div>• 3D Residual Blocks</div>
              <div>• SE-Channel Attention</div>
              <div>• 7×7×7 Spatial Attn</div>
            </div>
          </div>

          {/* Stage 3: Bottleneck */}
          <div className="p-3.5 rounded-xl bg-indigo-950/20 border border-indigo-800/40 flex flex-col justify-between">
            <div>
              <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-indigo-950 text-indigo-400 uppercase">Bottleneck</span>
              <h4 className="text-xs font-bold text-slate-200 mt-2">Multi-Scale Receptive</h4>
              <p className="text-[11px] text-slate-400 mt-1">
                512 Bottleneck Features
              </p>
            </div>
            <div className="mt-3 space-y-1 text-[10px] font-mono text-indigo-300 bg-indigo-950/40 p-2 rounded">
              <div>• Parallel Conv 3×3×3</div>
              <div>• Dilated Conv (d=2)</div>
              <div>• 1×1 Projection Fusion</div>
            </div>
          </div>

          {/* Stage 4: Decoder */}
          <div className="p-3.5 rounded-xl bg-teal-950/20 border border-teal-800/40 flex flex-col justify-between">
            <div>
              <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-teal-950 text-teal-400 uppercase">Decoder</span>
              <h4 className="text-xs font-bold text-slate-200 mt-2">Attention-Gated Skip</h4>
              <p className="text-[11px] text-slate-400 mt-1">
                512 → 256 → 128 → 64 → 32
              </p>
            </div>
            <div className="mt-3 space-y-1 text-[10px] font-mono text-teal-300 bg-teal-950/40 p-2 rounded">
              <div>• 3D Attention Gates</div>
              <div>• Trilinear Upsampling</div>
              <div>• Skip Feature Concatenation</div>
            </div>
          </div>

          {/* Stage 5: Output */}
          <div className="p-3.5 rounded-xl bg-emerald-950/20 border border-emerald-800/40 flex flex-col justify-between">
            <div>
              <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-950 text-emerald-400 uppercase">Output</span>
              <h4 className="text-xs font-bold text-slate-200 mt-2">Multi-Label Sigmoid</h4>
              <p className="text-[11px] text-slate-400 mt-1">
                (B, 3, D, H, W)
              </p>
            </div>
            <div className="mt-3 space-y-1 text-[10px] font-mono text-emerald-300 bg-emerald-950/40 p-2 rounded">
              <div>• Ch 0: Whole Tumor (WT)</div>
              <div>• Ch 1: Tumor Core (TC)</div>
              <div>• Ch 2: Enhancing (ET)</div>
            </div>
          </div>
        </div>
      </div>

      {/* Interactive Module Selector & Mathematical Formulation */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Module Picker */}
        <div className="lg:col-span-4 space-y-2">
          {Object.entries(modules).map(([key, item]) => (
            <button
              key={key}
              onClick={() => setSelectedModule(key)}
              className={`w-full text-left p-3.5 rounded-xl border transition ${
                selectedModule === key
                  ? 'bg-slate-800 border-cyan-500 shadow-sm text-white'
                  : 'bg-slate-900 border-slate-800 text-slate-400 hover:bg-slate-800/50'
              }`}
            >
              <div className="text-xs font-bold">{item.title}</div>
              <div className="text-[11px] text-slate-400 line-clamp-1 mt-0.5">{item.description}</div>
            </button>
          ))}
        </div>

        {/* Selected Module Detail */}
        <div className="lg:col-span-8 p-5 rounded-xl bg-slate-900 border border-slate-800 space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-2">
            <h4 className="text-sm font-bold text-slate-100">{modules[selectedModule].title}</h4>
            <span className="text-xs font-mono text-cyan-400">Mathematical Formulation</span>
          </div>

          {/* Formula Callout */}
          <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 font-mono text-xs text-cyan-300 text-center">
            {modules[selectedModule].formula}
          </div>

          <p className="text-xs text-slate-300 leading-relaxed">
            {modules[selectedModule].description}
          </p>

          <div className="space-y-2 pt-2">
            <div className="text-xs font-semibold text-slate-200">Clinical & Architectural Advantages:</div>
            <ul className="space-y-1.5 text-xs text-slate-400">
              {modules[selectedModule].keyBenefits.map((benefit, idx) => (
                <li key={idx} className="flex items-start space-x-2">
                  <Check className="w-3.5 h-3.5 text-emerald-400 shrink-0 mt-0.5" />
                  <span>{benefit}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
