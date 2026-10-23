import React from 'react';
import { Activity, Brain, Layers, BarChart3, Info, Database, CheckCircle2, AlertCircle } from 'lucide-react';

export default function Navbar({ activeTab, setActiveTab, healthData }) {
  const tabs = [
    { id: 'segmentation', label: 'MRI Segmentation & 3D Viewer', icon: Brain },
    { id: 'metrics', label: 'Performance & Metrics', icon: BarChart3 },
    { id: 'architecture', label: 'Improved Architecture', icon: Layers },
    { id: 'datasets', label: 'BraTS Datasets & Taxonomy', icon: Database },
    { id: 'system', label: 'System Health', icon: Activity },
  ];

  return (
    <header className="border-b border-slate-800 bg-slate-900/90 backdrop-blur sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-20">
          {/* Brand & Project Title */}
          <div className="flex items-center space-x-3">
            <div className="p-2.5 rounded-xl bg-gradient-to-tr from-cyan-600 to-indigo-600 text-white shadow-lg shadow-cyan-500/20">
              <Brain className="w-7 h-7 animate-pulse" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <h1 className="text-lg font-bold text-slate-100 tracking-tight leading-snug">
                  Multimodal MRI-Based Brain Tumor Segmentation
                </h1>
                <span className="hidden md:inline-flex px-2 py-0.5 text-xs font-semibold rounded-full bg-cyan-950 text-cyan-400 border border-cyan-800">
                  Attention Res-UNet
                </span>
              </div>
              <p className="text-xs text-slate-400">
                Using an Improved Attention-Based Deep Learning Framework • BraTS 2021 & BraTS 2023-MET
              </p>
            </div>
          </div>

          {/* System Status Pill */}
          <div className="hidden lg:flex items-center space-x-3">
            <div className="flex items-center space-x-2 px-3 py-1.5 rounded-full bg-slate-800/80 border border-slate-700 text-xs">
              {healthData?.status === 'healthy' ? (
                <>
                  <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping"></span>
                  <span className="w-2 h-2 rounded-full bg-emerald-500 -ml-3"></span>
                  <span className="text-emerald-400 font-medium">Backend Ready</span>
                  <span className="text-slate-500">•</span>
                  <span className="text-slate-300">PyTorch {healthData?.pytorch_version?.split('+')[0] || 'v2.14'}</span>
                  {healthData?.cuda_available ? (
                    <span className="text-cyan-400 font-semibold">(CUDA)</span>
                  ) : (
                    <span className="text-slate-400 font-medium">(CPU)</span>
                  )}
                </>
              ) : (
                <>
                  <span className="w-2 h-2 rounded-full bg-amber-400"></span>
                  <span className="text-amber-300 font-medium">Connecting...</span>
                </>
              )}
            </div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="flex space-x-1 overflow-x-auto py-2 scrollbar-none border-t border-slate-800/60">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center space-x-2 px-4 py-2 text-xs md:text-sm font-medium rounded-lg transition-all duration-150 whitespace-nowrap ${
                  isActive
                    ? 'bg-cyan-600/20 text-cyan-300 border border-cyan-500/30 shadow-sm'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                }`}
              >
                <Icon className={`w-4 h-4 ${isActive ? 'text-cyan-400' : 'text-slate-400'}`} />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
