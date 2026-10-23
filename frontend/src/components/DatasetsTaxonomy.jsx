import React, { useState, useEffect } from 'react';
import { Database, ExternalLink, ShieldAlert, CheckCircle2, BookOpen, DownloadCloud, FileCode } from 'lucide-react';

export default function DatasetsTaxonomy() {
  const [datasetInfo, setDatasetInfo] = useState(null);

  useEffect(() => {
    fetch('/api/datasets')
      .then((res) => res.json())
      .then((data) => setDatasetInfo(data))
      .catch((err) => console.error('Failed to fetch dataset info:', err));
  }, []);

  return (
    <div className="space-y-6">
      {/* Overview Banner */}
      <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div>
          <h2 className="text-lg font-bold text-white flex items-center space-x-2">
            <Database className="w-5 h-5 text-cyan-400" />
            <span>BraTS Datasets & Clinical Taxonomy</span>
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Cross-dataset evaluation: Adult Glioma (BraTS 2021) and Brain Metastases (BraTS 2023-MET)
          </p>
        </div>

        <div className="flex items-center space-x-2">
          <span className="px-3 py-1 rounded-full text-xs font-semibold bg-emerald-950 text-emerald-400 border border-emerald-800">
            Synapse Standardized
          </span>
        </div>
      </div>

      {/* Dataset Comparison Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* BraTS 2021 (Glioma) */}
        <div className="p-5 rounded-xl bg-slate-900 border border-slate-800 shadow-sm space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <div>
              <span className="text-[10px] font-bold text-cyan-400 uppercase tracking-wider">Primary Dataset</span>
              <h3 className="text-base font-bold text-white">BraTS 2021 (Glioma Challenge)</h3>
            </div>
            <a
              href="https://www.synapse.org/Synapse:syn25829067/wiki/610863"
              target="_blank"
              rel="noreferrer"
              className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-cyan-400 transition"
              title="Open Synapse Portal"
            >
              <ExternalLink className="w-4 h-4" />
            </a>
          </div>

          <div className="space-y-2 text-xs">
            <div className="flex justify-between py-1 border-b border-slate-800/60">
              <span className="text-slate-400">Pathology</span>
              <span className="text-slate-200 font-semibold">Adult Diffuse High/Low-Grade Glioma</span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-800/60">
              <span className="text-slate-400">Total Cases</span>
              <span className="text-slate-200 font-mono">1,251 mpMRI cases</span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-800/60">
              <span className="text-slate-400">Modalities</span>
              <span className="text-slate-200 font-mono">T1, T1ce, T2, FLAIR</span>
            </div>
          </div>

          {/* Raw Label Mapping */}
          <div className="space-y-2 pt-1">
            <h4 className="text-xs font-bold text-slate-300">Raw Annotation Labels:</h4>
            <div className="space-y-1.5 text-xs font-mono">
              <div className="p-2 rounded bg-slate-950 border border-slate-800 flex justify-between">
                <span>Label 0: Background</span>
                <span className="text-slate-500">Normal brain tissue</span>
              </div>
              <div className="p-2 rounded bg-slate-950 border border-slate-800 flex justify-between">
                <span className="text-cyan-400">Label 1: NCR</span>
                <span className="text-slate-400">Necrotic Tumor Core</span>
              </div>
              <div className="p-2 rounded bg-slate-950 border border-slate-800 flex justify-between">
                <span className="text-emerald-400">Label 2: ED</span>
                <span className="text-slate-400">Peritumoral Infiltrative Edema</span>
              </div>
              <div className="p-2 rounded bg-slate-950 border border-slate-800 flex justify-between">
                <span className="text-rose-400">Label 3: ET</span>
                <span className="text-slate-400">Enhancing Tumor Rim</span>
              </div>
            </div>
          </div>
        </div>

        {/* BraTS 2023-MET (Metastasis) */}
        <div className="p-5 rounded-xl bg-slate-900 border border-slate-800 shadow-sm space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <div>
              <span className="text-[10px] font-bold text-indigo-400 uppercase tracking-wider">Held-Out Test Dataset</span>
              <h3 className="text-base font-bold text-white">BraTS 2023-MET (Brain Metastasis)</h3>
            </div>
            <a
              href="https://www.synapse.org/Synapse:syn51156910/wiki/622351"
              target="_blank"
              rel="noreferrer"
              className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-indigo-400 transition"
              title="Open Synapse Portal"
            >
              <ExternalLink className="w-4 h-4" />
            </a>
          </div>

          <div className="space-y-2 text-xs">
            <div className="flex justify-between py-1 border-b border-slate-800/60">
              <span className="text-slate-400">Pathology</span>
              <span className="text-slate-200 font-semibold">Metastatic Brain Tumors (Multiple)</span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-800/60">
              <span className="text-slate-400">Total Cases</span>
              <span className="text-slate-200 font-mono">354 mpMRI cases</span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-800/60">
              <span className="text-slate-400">Modalities</span>
              <span className="text-slate-200 font-mono">T1, T1ce, T2, FLAIR</span>
            </div>
          </div>

          {/* Raw Label Mapping */}
          <div className="space-y-2 pt-1">
            <h4 className="text-xs font-bold text-slate-300">Raw Annotation Labels:</h4>
            <div className="space-y-1.5 text-xs font-mono">
              <div className="p-2 rounded bg-slate-950 border border-slate-800 flex justify-between">
                <span>Label 0: Background</span>
                <span className="text-slate-500">Normal tissue</span>
              </div>
              <div className="p-2 rounded bg-slate-950 border border-slate-800 flex justify-between">
                <span className="text-cyan-400">Label 1: NETC</span>
                <span className="text-slate-400">Non-Enhancing Tumor Core</span>
              </div>
              <div className="p-2 rounded bg-slate-950 border border-slate-800 flex justify-between">
                <span className="text-indigo-400">Label 2: SNFH</span>
                <span className="text-slate-400">Surrounding FLAIR Hyperintensity</span>
              </div>
              <div className="p-2 rounded bg-slate-950 border border-slate-800 flex justify-between">
                <span className="text-rose-400">Label 3: ET</span>
                <span className="text-slate-400">Enhancing Tumor Focus</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Clinical Taxonomy Matrix */}
      <div className="p-5 rounded-xl bg-slate-900 border border-slate-800 shadow-sm space-y-4">
        <h3 className="text-sm font-semibold text-slate-200">
          Clinical Taxonomy & Hierarchical Subregion Definitions
        </h3>
        
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400 uppercase">
                <th className="py-2.5 px-3">Hierarchical Region</th>
                <th className="py-2.5 px-3">BraTS 2021 (Glioma)</th>
                <th className="py-2.5 px-3">BraTS 2023-MET (Metastasis)</th>
                <th className="py-2.5 px-3">Clinical Significance</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 text-slate-300">
              <tr>
                <td className="py-2.5 px-3 font-semibold text-emerald-400">Whole Tumor (WT)</td>
                <td className="py-2.5 px-3 font-mono">Labels 1 + 2 + 3 (NCR + ED + ET)</td>
                <td className="py-2.5 px-3 font-mono">Labels 1 + 2 + 3 (NETC + SNFH + ET)</td>
                <td className="py-2.5 px-3 text-slate-400">Gross total tumor footprint including peritumoral edema</td>
              </tr>
              <tr>
                <td className="py-2.5 px-3 font-semibold text-cyan-400">Tumor Core (TC)</td>
                <td className="py-2.5 px-3 font-mono">Labels 1 + 3 (NCR + ET)</td>
                <td className="py-2.5 px-3 font-mono">Labels 1 + 3 (NETC + ET)</td>
                <td className="py-2.5 px-3 text-slate-400">Resectable mass excluding surrounding edema/SNFH</td>
              </tr>
              <tr>
                <td className="py-2.5 px-3 font-semibold text-rose-400">Enhancing Tumor (ET)</td>
                <td className="py-2.5 px-3 font-mono">Label 3 (ET)</td>
                <td className="py-2.5 px-3 font-mono">Label 3 (ET)</td>
                <td className="py-2.5 px-3 text-slate-400">Hypervascular active tumor; target for radiotherapy</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Dataset Download & Setup Guide */}
      <div className="p-5 rounded-xl bg-slate-900 border border-slate-800 shadow-sm space-y-3">
        <h3 className="text-sm font-semibold text-slate-200 flex items-center space-x-2">
          <DownloadCloud className="w-4 h-4 text-cyan-400" />
          <span>Setting Up Real BraTS Challenge Datasets</span>
        </h3>
        <p className="text-xs text-slate-300 leading-relaxed">
          Both BraTS datasets are hosted on Synapse.org under medical data use agreements. To evaluate on the full original cohorts:
        </p>
        <ol className="list-decimal list-inside text-xs text-slate-400 space-y-1.5 pl-1">
          <li>Register an account at <a href="https://www.synapse.org" target="_blank" rel="noreferrer" className="text-cyan-400 underline">Synapse.org</a>.</li>
          <li>Accept the Data Use Agreement on Synapse:syn25829067 (BraTS 2021) and Synapse:syn51156910 (BraTS 2023-MET).</li>
          <li>Extract cases to <code className="text-cyan-300 bg-slate-950 px-1.5 py-0.5 rounded">./data/BraTS2021/</code> and <code className="text-cyan-300 bg-slate-950 px-1.5 py-0.5 rounded">./data/BraTS2023_MET/</code>.</li>
          <li>Run <code className="text-cyan-300 bg-slate-950 px-1.5 py-0.5 rounded">python evaluation/evaluate.py --dataset brats2021</code>.</li>
        </ol>
      </div>
    </div>
  );
}
