import React, { useState, useEffect, useRef } from 'react';
import { 
  Upload, Play, Eye, EyeOff, Sliders, Layers, RefreshCw, 
  CheckCircle, AlertTriangle, FileText, ChevronRight, BarChart, 
  HelpCircle, Sparkles, Activity
} from 'lucide-react';

export default function SegmentationViewer() {
  // State for files
  const [t1File, setT1File] = useState(null);
  const [t1ceFile, setT1ceFile] = useState(null);
  const [t2File, setT2File] = useState(null);
  const [flairFile, setFlairFile] = useState(null);
  const [patientId, setPatientId] = useState('BraTS2021_Demo_01');
  const [modelType, setModelType] = useState('improved');

  // Viewer state
  const [plane, setPlane] = useState('axial'); // axial, sagittal, coronal
  const [sliceIndex, setSliceIndex] = useState(32);
  const totalSlices = 64;
  const [modality, setModality] = useState('flair'); // t1, t1ce, t2, flair
  const [showMask, setShowMask] = useState(true);
  const [maskOpacity, setMaskOpacity] = useState(0.65);
  const [activeRegions, setActiveRegions] = useState({ WT: true, TC: true, ET: true });

  // Processing state
  const [isLoading, setIsLoading] = useState(false);
  const [predictionResult, setPredictionResult] = useState(null);
  const [errorMsg, setErrorMsg] = useState('');
  const [isDemoActive, setIsDemoActive] = useState(true);

  const canvasRef = useRef(null);

  // Initial demo load
  useEffect(() => {
    loadDemoPatient();
  }, []);

  // Re-draw slice canvas whenever slice, modality, plane, mask, or opacity changes
  useEffect(() => {
    drawSlice();
  }, [sliceIndex, modality, plane, showMask, maskOpacity, activeRegions, predictionResult]);

  const loadDemoPatient = async () => {
    setIsLoading(true);
    setErrorMsg('');
    try {
      // Simulate/trigger prediction using demo patient metadata
      setTimeout(() => {
        setPredictionResult({
          patient_id: 'BraTS2021_00001_Demo',
          model_type: modelType,
          model_loaded: true,
          checkpoint_used: 'checkpoints/best_model.pth',
          inference_time_seconds: 0.884,
          regions: {
            WT: { voxel_count: 9072, volume_mm3: 9072.0, percentage_of_brain: 3.46 },
            TC: { voxel_count: 2367, volume_mm3: 2367.0, percentage_of_brain: 0.90 },
            ET: { voxel_count: 1420, volume_mm3: 1420.0, percentage_of_brain: 0.54 },
          },
          output_shape: [64, 64, 64],
          message: 'Segmentation completed successfully using Improved Attention-Based Deep Learning Framework.',
        });
        setIsDemoActive(true);
        setIsLoading(false);
      }, 400);
    } catch (err) {
      setErrorMsg('Failed to load demo data: ' + err.message);
      setIsLoading(false);
    }
  };

  const handleFileUpload = async (e) => {
    e.preventDefault();
    if (!t1File || !t1ceFile || !t2File || !flairFile) {
      setErrorMsg('Please select all four MRI modalities (T1, T1ce, T2, FLAIR) or use the Demo Patient.');
      return;
    }

    setIsLoading(true);
    setErrorMsg('');
    const formData = new FormData();
    formData.append('t1_file', t1File);
    formData.append('t1ce_file', t1ceFile);
    formData.append('t2_file', t2File);
    formData.append('flair_file', flairFile);
    formData.append('patient_id', patientId || 'Patient_Upload');
    formData.append('model_type', modelType);

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || 'Segmentation failed');
      }

      const data = await response.json();
      setPredictionResult(data);
      setIsDemoActive(false);
    } catch (err) {
      setErrorMsg(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  // Synthetic Slice Drawing Engine for Medical Viewer
  const drawSlice = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    // Clear background
    ctx.fillStyle = '#050811';
    ctx.fillRect(0, 0, width, height);

    // Coordinate normalization (-1 to 1)
    const normSlice = (sliceIndex / (totalSlices - 1)) * 2 - 1; // -1 to 1

    const imgData = ctx.createImageData(width, height);
    const data = imgData.data;

    // Brain ellipse parameters
    const cx = width / 2;
    const cy = height / 2;
    const rx = width * 0.38 * Math.sqrt(Math.max(0, 1 - normSlice * normSlice * 0.8));
    const ry = height * 0.44 * Math.sqrt(Math.max(0, 1 - normSlice * normSlice * 0.8));

    // Tumor position in 3D
    const tumorCenterX = cx + width * 0.12;
    const tumorCenterY = cy - height * 0.08;
    const tumorSliceDist = Math.abs(normSlice - 0.0); // tumor centered at middle slice
    const tumorActiveRadius = Math.max(0, 1 - (tumorSliceDist / 0.5) ** 2);

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const pixelIdx = (y * width + x) * 4;

        // Check if inside brain skull outline
        const dx = (x - cx) / rx;
        const dy = (y - cy) / ry;
        const distSq = dx * dx + dy * dy;

        if (distSq > 1.0) {
          // Outside brain
          data[pixelIdx] = 5;
          data[pixelIdx + 1] = 8;
          data[pixelIdx + 2] = 17;
          data[pixelIdx + 3] = 255;
          continue;
        }

        // Inside brain tissue: base gray matter + white matter gradient
        const brainGrad = Math.max(0, 1.0 - Math.sqrt(distSq));
        let intensity = 40 + brainGrad * 120;

        // Modality-specific normal tissue contrast
        if (modality === 't1') {
          intensity = 50 + brainGrad * 130;
        } else if (modality === 't1ce') {
          intensity = 45 + brainGrad * 135;
        } else if (modality === 't2') {
          intensity = 35 + brainGrad * 110;
        } else if (modality === 'flair') {
          intensity = 30 + brainGrad * 115;
        }

        // Add subtle anatomical texture
        const noise = ((Math.sin(x * 0.25) * Math.cos(y * 0.25) + 1) / 2) * 15;
        intensity += noise;

        // Tumor region calculation
        const tdx = (x - tumorCenterX) / (width * 0.14);
        const tdy = (y - tumorCenterY) / (height * 0.14);
        const tDist = Math.sqrt(tdx * tdx + tdy * tdy);

        let isWT = false;
        let isTC = false;
        let isET = false;

        if (tumorActiveRadius > 0 && tDist < tumorActiveRadius) {
          // Inside tumor
          if (tDist < tumorActiveRadius * 0.4) {
            isTC = true; // Necrotic Core (NCR)
          } else if (tDist < tumorActiveRadius * 0.75) {
            isET = true; // Enhancing Tumor (ET)
            isTC = true; // TC includes ET + NCR
          } else {
            isWT = true; // Edema (ED)
          }
          isWT = true; // WT includes all

          // Modality appearance alteration
          if (modality === 't1') {
            intensity = Math.max(20, intensity - 45); // hypointense
          } else if (modality === 't1ce') {
            if (isET) intensity = Math.min(255, intensity + 110); // bright rim
            else if (isTC) intensity = Math.max(25, intensity - 30);
          } else if (modality === 't2') {
            if (isWT) intensity = Math.min(255, intensity + 85); // hyperintense edema
          } else if (modality === 'flair') {
            if (isWT) intensity = Math.min(255, intensity + 105); // high hyperintensity
          }
        }

        // Base grayscale pixel
        let r = intensity;
        let g = intensity;
        let b = intensity;

        // Apply Segmentation Mask Overlays if enabled
        if (showMask && tumorActiveRadius > 0) {
          const alpha = maskOpacity;
          if (isET && activeRegions.ET) {
            // Enhancing Tumor: Red / Coral
            r = r * (1 - alpha) + 239 * alpha;
            g = g * (1 - alpha) + 68 * alpha;
            b = b * (1 - alpha) + 68 * alpha;
          } else if (isTC && activeRegions.TC) {
            // Tumor Core: Cyan / Blue
            r = r * (1 - alpha) + 6 * alpha;
            g = g * (1 - alpha) + 182 * alpha;
            b = b * (1 - alpha) + 212 * alpha;
          } else if (isWT && activeRegions.WT) {
            // Edema / Whole Tumor: Emerald / Green
            r = r * (1 - alpha) + 34 * alpha;
            g = g * (1 - alpha) + 197 * alpha;
            b = b * (1 - alpha) + 94 * alpha;
          }
        }

        data[pixelIdx] = Math.min(255, Math.max(0, r));
        data[pixelIdx + 1] = Math.min(255, Math.max(0, g));
        data[pixelIdx + 2] = Math.min(255, Math.max(0, b));
        data[pixelIdx + 3] = 255;
      }
    }

    ctx.putImageData(imgData, 0, 0);

    // Draw Crosshairs / Medical orientation annotations
    ctx.strokeStyle = '#334155';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);

    // Center cross
    ctx.beginPath();
    ctx.moveTo(width / 2, 8);
    ctx.lineTo(width / 2, 24);
    ctx.moveTo(width / 2, height - 24);
    ctx.lineTo(width / 2, height - 8);
    ctx.moveTo(8, height / 2);
    ctx.lineTo(24, height / 2);
    ctx.moveTo(width - 24, height / 2);
    ctx.lineTo(width - 8, height / 2);
    ctx.stroke();
    ctx.setLineDash([]);

    // Medical Orientation Labels (A, P, L, R)
    ctx.fillStyle = '#94a3b8';
    ctx.font = '10px Inter, monospace';
    if (plane === 'axial') {
      ctx.fillText('A (Anterior)', width / 2 - 28, 20);
      ctx.fillText('P (Posterior)', width / 2 - 30, height - 12);
      ctx.fillText('R', 12, height / 2 + 4);
      ctx.fillText('L', width - 20, height / 2 + 4);
    } else if (plane === 'coronal') {
      ctx.fillText('S (Superior)', width / 2 - 28, 20);
      ctx.fillText('I (Inferior)', width / 2 - 26, height - 12);
      ctx.fillText('R', 12, height / 2 + 4);
      ctx.fillText('L', width - 20, height / 2 + 4);
    } else {
      ctx.fillText('S (Superior)', width / 2 - 28, 20);
      ctx.fillText('I (Inferior)', width / 2 - 26, height - 12);
      ctx.fillText('A', 12, height / 2 + 4);
      ctx.fillText('P', width - 20, height / 2 + 4);
    }
  };

  return (
    <div className="space-y-6">
      {/* Top Banner with Quick Sample Load */}
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between p-4 rounded-xl bg-slate-900 border border-slate-800 shadow-sm gap-4">
        <div>
          <h2 className="text-lg font-bold text-white flex items-center space-x-2">
            <span>Interactive Multimodal MRI Tumor Segmentation</span>
            <span className="px-2 py-0.5 text-xs font-semibold rounded bg-cyan-950 text-cyan-400 border border-cyan-800">
              3D Volume
            </span>
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Input: T1, T1ce, T2, FLAIR • Output: Multi-label segmentation (WT, TC, ET) with volume calculation
          </p>
        </div>
        <div className="flex items-center space-x-3 w-full md:w-auto">
          <button
            onClick={loadDemoPatient}
            className="flex-1 md:flex-initial flex items-center justify-center space-x-2 px-4 py-2 text-xs font-semibold rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white transition shadow-lg shadow-cyan-600/20"
          >
            <Sparkles className="w-3.5 h-3.5" />
            <span>Load Demo Patient</span>
          </button>
        </div>
      </div>

      {errorMsg && (
        <div className="p-3.5 rounded-lg bg-rose-950/50 border border-rose-800 text-rose-300 text-xs flex items-center space-x-2">
          <AlertTriangle className="w-4 h-4 shrink-0" />
          <span>{errorMsg}</span>
        </div>
      )}

      {/* Main Grid: Upload & Settings (Left) | Interactive 3D Slice Viewer (Right) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Upload & Controls */}
        <div className="lg:col-span-5 space-y-5">
          {/* File Upload Box */}
          <div className="p-5 rounded-xl bg-slate-900 border border-slate-800 shadow-sm space-y-4">
            <h3 className="text-sm font-semibold text-slate-200 flex items-center space-x-2">
              <Upload className="w-4 h-4 text-cyan-400" />
              <span>Multimodal NIfTI Upload (.nii / .nii.gz)</span>
            </h3>

            <div className="grid grid-cols-2 gap-3 text-xs">
              {/* T1 */}
              <div className="space-y-1">
                <label className="font-medium text-slate-300 flex items-center justify-between">
                  <span>1. T1-Weighted</span>
                  <span className="text-[10px] text-slate-500">Anatomy</span>
                </label>
                <input
                  type="file"
                  accept=".nii,.gz"
                  onChange={(e) => setT1File(e.target.files[0])}
                  className="w-full text-slate-400 file:mr-2 file:py-1.5 file:px-2.5 file:rounded file:border-0 file:text-[11px] file:bg-slate-800 file:text-cyan-400 hover:file:bg-slate-700 cursor-pointer bg-slate-950 p-1 rounded border border-slate-800 text-[11px]"
                />
              </div>

              {/* T1ce */}
              <div className="space-y-1">
                <label className="font-medium text-slate-300 flex items-center justify-between">
                  <span>2. T1ce (Gadolinium)</span>
                  <span className="text-[10px] text-rose-400">Enhancing</span>
                </label>
                <input
                  type="file"
                  accept=".nii,.gz"
                  onChange={(e) => setT1ceFile(e.target.files[0])}
                  className="w-full text-slate-400 file:mr-2 file:py-1.5 file:px-2.5 file:rounded file:border-0 file:text-[11px] file:bg-slate-800 file:text-cyan-400 hover:file:bg-slate-700 cursor-pointer bg-slate-950 p-1 rounded border border-slate-800 text-[11px]"
                />
              </div>

              {/* T2 */}
              <div className="space-y-1">
                <label className="font-medium text-slate-300 flex items-center justify-between">
                  <span>3. T2-Weighted</span>
                  <span className="text-[10px] text-emerald-400">Edema</span>
                </label>
                <input
                  type="file"
                  accept=".nii,.gz"
                  onChange={(e) => setT2File(e.target.files[0])}
                  className="w-full text-slate-400 file:mr-2 file:py-1.5 file:px-2.5 file:rounded file:border-0 file:text-[11px] file:bg-slate-800 file:text-cyan-400 hover:file:bg-slate-700 cursor-pointer bg-slate-950 p-1 rounded border border-slate-800 text-[11px]"
                />
              </div>

              {/* FLAIR */}
              <div className="space-y-1">
                <label className="font-medium text-slate-300 flex items-center justify-between">
                  <span>4. FLAIR</span>
                  <span className="text-[10px] text-cyan-400">Hyperintensity</span>
                </label>
                <input
                  type="file"
                  accept=".nii,.gz"
                  onChange={(e) => setFlairFile(e.target.files[0])}
                  className="w-full text-slate-400 file:mr-2 file:py-1.5 file:px-2.5 file:rounded file:border-0 file:text-[11px] file:bg-slate-800 file:text-cyan-400 hover:file:bg-slate-700 cursor-pointer bg-slate-950 p-1 rounded border border-slate-800 text-[11px]"
                />
              </div>
            </div>

            {/* Model Architecture & Patient ID */}
            <div className="grid grid-cols-2 gap-3 pt-2">
              <div className="space-y-1">
                <label className="text-xs font-medium text-slate-300">Model Architecture</label>
                <select
                  value={modelType}
                  onChange={(e) => setModelType(e.target.value)}
                  className="w-full px-2.5 py-1.5 rounded-lg bg-slate-950 border border-slate-800 text-xs text-slate-200 focus:outline-none focus:border-cyan-500"
                >
                  <option value="improved">Proposed Attention Res-UNet (Improved)</option>
                  <option value="baseline">Standard 3D U-Net (Baseline)</option>
                </select>
              </div>
              <div className="space-y-1">
                <label className="text-xs font-medium text-slate-300">Patient Identifier</label>
                <input
                  type="text"
                  value={patientId}
                  onChange={(e) => setPatientId(e.target.value)}
                  placeholder="e.g. BraTS2021_00001"
                  className="w-full px-2.5 py-1.5 rounded-lg bg-slate-950 border border-slate-800 text-xs text-slate-200 focus:outline-none focus:border-cyan-500"
                />
              </div>
            </div>

            <button
              onClick={handleFileUpload}
              disabled={isLoading}
              className="w-full flex items-center justify-center space-x-2 py-2.5 px-4 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white font-medium text-xs transition duration-150 disabled:opacity-50"
            >
              {isLoading ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  <span>Segmenting 3D Volume...</span>
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 fill-current" />
                  <span>Execute 3D Multimodal Segmentation</span>
                </>
              )}
            </button>
          </div>

          {/* Quantitative Volumetric Analysis */}
          {predictionResult && (
            <div className="p-5 rounded-xl bg-slate-900 border border-slate-800 shadow-sm space-y-4">
              <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                <h3 className="text-sm font-semibold text-slate-200 flex items-center space-x-2">
                  <BarChart className="w-4 h-4 text-emerald-400" />
                  <span>Volumetric Quantification</span>
                </h3>
                <span className="text-[11px] text-slate-400">
                  Latency: <strong className="text-emerald-400">{predictionResult.inference_time_seconds}s</strong>
                </span>
              </div>

              {/* Subregion breakdown cards */}
              <div className="grid grid-cols-3 gap-2.5 text-center">
                {/* Whole Tumor (WT) */}
                <div className="p-2.5 rounded-lg bg-emerald-950/30 border border-emerald-900/50">
                  <div className="text-[10px] font-bold text-emerald-400 tracking-wider">WHOLE TUMOR (WT)</div>
                  <div className="text-base font-extrabold text-slate-100 mt-1">
                    {predictionResult.regions?.WT?.volume_mm3?.toLocaleString() || 0}
                    <span className="text-[10px] font-normal text-slate-400 ml-1">mm³</span>
                  </div>
                  <div className="text-[10px] text-slate-400 mt-0.5">
                    {predictionResult.regions?.WT?.percentage_of_brain || 0}% Brain
                  </div>
                  <div className="text-[9px] text-emerald-500/80 mt-1">NCR + ED + ET</div>
                </div>

                {/* Tumor Core (TC) */}
                <div className="p-2.5 rounded-lg bg-cyan-950/30 border border-cyan-900/50">
                  <div className="text-[10px] font-bold text-cyan-400 tracking-wider">TUMOR CORE (TC)</div>
                  <div className="text-base font-extrabold text-slate-100 mt-1">
                    {predictionResult.regions?.TC?.volume_mm3?.toLocaleString() || 0}
                    <span className="text-[10px] font-normal text-slate-400 ml-1">mm³</span>
                  </div>
                  <div className="text-[10px] text-slate-400 mt-0.5">
                    {predictionResult.regions?.TC?.percentage_of_brain || 0}% Brain
                  </div>
                  <div className="text-[9px] text-cyan-500/80 mt-1">NCR + ET</div>
                </div>

                {/* Enhancing Tumor (ET) */}
                <div className="p-2.5 rounded-lg bg-rose-950/30 border border-rose-900/50">
                  <div className="text-[10px] font-bold text-rose-400 tracking-wider">ENHANCING (ET)</div>
                  <div className="text-base font-extrabold text-slate-100 mt-1">
                    {predictionResult.regions?.ET?.volume_mm3?.toLocaleString() || 0}
                    <span className="text-[10px] font-normal text-slate-400 ml-1">mm³</span>
                  </div>
                  <div className="text-[10px] text-slate-400 mt-0.5">
                    {predictionResult.regions?.ET?.percentage_of_brain || 0}% Brain
                  </div>
                  <div className="text-[9px] text-rose-500/80 mt-1">Active Rim</div>
                </div>
              </div>

              {/* Model Verification Tag */}
              <div className="flex items-center space-x-2 text-[11px] text-slate-400 pt-1">
                <CheckCircle className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
                <span>Checkpoint: <code className="text-slate-300 font-mono text-[10px]">{predictionResult.checkpoint_used || 'best_model.pth'}</code></span>
              </div>
            </div>
          )}
        </div>

        {/* Right Column: Interactive Slice Viewer Canvas */}
        <div className="lg:col-span-7 space-y-4">
          <div className="p-5 rounded-xl bg-slate-900 border border-slate-800 shadow-sm space-y-4">
            {/* Viewer Header Toolbar */}
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-3">
              {/* Plane Selector */}
              <div className="flex items-center space-x-1 bg-slate-950 p-1 rounded-lg border border-slate-800">
                {['axial', 'coronal', 'sagittal'].map((p) => (
                  <button
                    key={p}
                    onClick={() => setPlane(p)}
                    className={`px-3 py-1 rounded text-xs font-semibold capitalize transition ${
                      plane === p ? 'bg-cyan-600 text-white shadow-sm' : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    {p}
                  </button>
                ))}
              </div>

              {/* Modality Selector */}
              <div className="flex items-center space-x-1 bg-slate-950 p-1 rounded-lg border border-slate-800">
                {['t1', 't1ce', 't2', 'flair'].map((m) => (
                  <button
                    key={m}
                    onClick={() => setModality(m)}
                    className={`px-2.5 py-1 rounded text-xs font-semibold uppercase transition ${
                      modality === m ? 'bg-indigo-600 text-white shadow-sm' : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    {m}
                  </button>
                ))}
              </div>

              {/* Overlay Toggle */}
              <button
                onClick={() => setShowMask(!showMask)}
                className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border transition ${
                  showMask 
                    ? 'bg-emerald-950/60 text-emerald-400 border-emerald-800' 
                    : 'bg-slate-950 text-slate-400 border-slate-800'
                }`}
              >
                {showMask ? <Eye className="w-3.5 h-3.5" /> : <EyeOff className="w-3.5 h-3.5" />}
                <span>Mask Overlay</span>
              </button>
            </div>

            {/* Canvas Display Viewport */}
            <div className="relative flex items-center justify-center bg-black rounded-lg border border-slate-800/80 p-2 overflow-hidden shadow-inner aspect-square max-w-[480px] mx-auto">
              <canvas
                ref={canvasRef}
                width={360}
                height={360}
                className="rounded border border-slate-800 shadow-2xl cursor-crosshair"
              />

              {/* Top-Right Slice Indicator Badge */}
              <div className="absolute top-4 right-4 px-2.5 py-1 rounded bg-slate-900/85 backdrop-blur border border-slate-800 text-[11px] font-mono text-cyan-400 shadow">
                Slice {sliceIndex + 1} / {totalSlices}
              </div>

              {/* Bottom-Left Patient & Plane Badge */}
              <div className="absolute bottom-4 left-4 px-2.5 py-1 rounded bg-slate-900/85 backdrop-blur border border-slate-800 text-[10px] text-slate-300 shadow">
                {predictionResult?.patient_id || 'Sample_Patient'} • <span className="capitalize font-semibold text-cyan-300">{plane}</span> ({modality.toUpperCase()})
              </div>
            </div>

            {/* Interactive Slider Controls */}
            <div className="space-y-3 pt-2">
              {/* Slice Slider */}
              <div className="flex items-center space-x-3">
                <span className="text-xs font-medium text-slate-400 w-16">Slice Z:</span>
                <input
                  type="range"
                  min={0}
                  max={totalSlices - 1}
                  value={sliceIndex}
                  onChange={(e) => setSliceIndex(parseInt(e.target.value))}
                  className="flex-1 h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-500"
                />
                <span className="text-xs font-mono text-slate-300 w-12 text-right">
                  {sliceIndex}
                </span>
              </div>

              {/* Mask Opacity Slider */}
              <div className="flex items-center space-x-3">
                <span className="text-xs font-medium text-slate-400 w-16">Opacity:</span>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={maskOpacity}
                  onChange={(e) => setMaskOpacity(parseFloat(e.target.value))}
                  className="flex-1 h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-indigo-500"
                />
                <span className="text-xs font-mono text-slate-300 w-12 text-right">
                  {Math.round(maskOpacity * 100)}%
                </span>
              </div>

              {/* Subregion Color Legend & Individual Toggles */}
              <div className="flex flex-wrap items-center justify-between pt-2 border-t border-slate-800/80 text-xs">
                <span className="text-slate-400 font-medium">Subregions:</span>
                <div className="flex items-center space-x-3">
                  <label className="flex items-center space-x-1.5 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={activeRegions.WT}
                      onChange={(e) => setActiveRegions({ ...activeRegions, WT: e.target.checked })}
                      className="rounded bg-slate-800 border-slate-700 text-emerald-500 focus:ring-0"
                    />
                    <span className="w-2.5 h-2.5 rounded-full bg-emerald-500"></span>
                    <span className="text-slate-300">WT (Edema)</span>
                  </label>

                  <label className="flex items-center space-x-1.5 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={activeRegions.TC}
                      onChange={(e) => setActiveRegions({ ...activeRegions, TC: e.target.checked })}
                      className="rounded bg-slate-800 border-slate-700 text-cyan-500 focus:ring-0"
                    />
                    <span className="w-2.5 h-2.5 rounded-full bg-cyan-500"></span>
                    <span className="text-slate-300">TC (Core)</span>
                  </label>

                  <label className="flex items-center space-x-1.5 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={activeRegions.ET}
                      onChange={(e) => setActiveRegions({ ...activeRegions, ET: e.target.checked })}
                      className="rounded bg-slate-800 border-slate-700 text-rose-500 focus:ring-0"
                    />
                    <span className="w-2.5 h-2.5 rounded-full bg-rose-500"></span>
                    <span className="text-slate-300">ET (Enhancing)</span>
                  </label>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
