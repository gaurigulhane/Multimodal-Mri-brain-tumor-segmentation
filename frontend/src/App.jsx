import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import SegmentationViewer from './components/SegmentationViewer';
import MetricsDashboard from './components/MetricsDashboard';
import ArchitectureExplorer from './components/ArchitectureExplorer';
import DatasetsTaxonomy from './components/DatasetsTaxonomy';
import SystemStatus from './components/SystemStatus';

export default function App() {
  const [activeTab, setActiveTab] = useState('segmentation');
  const [healthData, setHealthData] = useState(null);

  useEffect(() => {
    fetchHealth();
  }, []);

  const fetchHealth = async () => {
    try {
      const res = await fetch('/api/health');
      if (res.ok) {
        const data = await res.json();
        setHealthData(data);
      }
    } catch (err) {
      console.warn('Backend not yet reachable:', err);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      {/* Navigation Header */}
      <Navbar 
        activeTab={activeTab} 
        setActiveTab={setActiveTab} 
        healthData={healthData} 
      />

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {activeTab === 'segmentation' && <SegmentationViewer />}
        {activeTab === 'metrics' && <MetricsDashboard />}
        {activeTab === 'architecture' && <ArchitectureExplorer />}
        {activeTab === 'datasets' && <DatasetsTaxonomy />}
        {activeTab === 'system' && <SystemStatus healthData={healthData} />}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-900 bg-slate-950 py-6 text-center text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 space-y-1">
          <p className="font-medium text-slate-400">
            Multimodal MRI-Based Brain Tumor Segmentation Using an Improved Attention-Based Deep Learning Framework
          </p>
          <p className="text-[11px] text-slate-600">
            Final-Year AI/ML Research & Engineering Project • PyTorch 2.14 • MONAI 1.6 • BraTS 2021 & BraTS 2023-MET
          </p>
        </div>
      </footer>
    </div>
  );
}
