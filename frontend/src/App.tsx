import React, { useState } from 'react';
import { Sliders, Activity, RefreshCw, AlertCircle } from 'lucide-react';

export default function App() {
  const [timeStep, setTimeStep] = useState<number>(100);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [metrics, setMetrics] = useState({ uLoss: 0.0012, pdeLoss: 0.0001 });

  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    try {
      // Endpoint call to your FastAPI backend
      const response = await fetch(`http://localhost:8000/predict/field?t=${timeStep}`);
      if (!response.ok) throw new Error("Inference service unreachable.");
      const data = await response.json();
      setMetrics(data);
    } catch (err: any) {
      setError(err.message || "Failed to fetch field prediction.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background text-foreground p-8 font-sans">
      <header className="flex justify-between items-center mb-8 border-b border-border pb-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">PINNflow // Fluid Dynamics Engine</h1>
          <p className="text-sm text-slate-400">Real-time Navier-Stokes Neural Reconstruction Dashboard</p>
        </div>
        <div className="flex items-center gap-2 bg-slate-800 px-3 py-1.5 rounded-md border border-border">
          <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse"></span>
          <span className="text-xs font-medium">Model Active (Reynolds: 100)</span>
        </div>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Control Card */}
        <div className="bg-slate-900/50 border border-border rounded-xl p-6 shadow-xl flex flex-col justify-between">
          <div>
            <h2 className="text-lg font-semibold flex items-center gap-2 mb-4">
              <Sliders className="w-5 h-5 text-primary" /> Query Controls
            </h2>
            <div className="space-y-4">
              <div>
                <label className="text-xs font-medium text-slate-400 block mb-2">
                  Time Snapshot Index: {timeStep}
                </label>
                <input 
                  type="range" 
                  min="0" 
                  max="150" 
                  value={timeStep} 
                  onChange={(e) => setTimeStep(Number(e.target.value))}
                  className="w-full accent-primary bg-slate-800 rounded-lg h-2 cursor-pointer"
                />
              </div>
            </div>
          </div>

          <button 
            onClick={handlePredict}
            disabled={loading}
            className="mt-6 w-full bg-primary hover:bg-blue-600 text-primary-foreground font-medium py-2.5 rounded-lg transition-all flex items-center justify-center gap-2 disabled:opacity-50"
          >
            {loading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Activity className="w-4 h-4" />}
            {loading ? "Computing Inference..." : "Run Field Prediction"}
          </button>
        </div>

        {/* Visualization Canvas Box */}
        <div className="md:col-span-2 bg-slate-900/50 border border-border rounded-xl p-6 shadow-xl flex flex-col justify-between">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-sm font-semibold tracking-wider text-slate-300 uppercase">Flow Reconstruction View</h3>
            <span className="text-xs text-slate-500">Delaunay Mesh Render</span>
          </div>

          <div className="flex-1 bg-slate-950 rounded-lg border border-border flex items-center justify-center relative min-h-[300px]">
            {loading ? (
              <div className="flex flex-col items-center gap-2 text-slate-400">
                <RefreshCw className="w-8 h-8 animate-spin text-primary" />
                <p className="text-xs">Evaluating network forward pass...</p>
              </div>
            ) : error ? (
              <div className="flex flex-col items-center gap-2 text-danger">
                <AlertCircle className="w-8 h-8" />
                <p className="text-xs">{error}</p>
              </div>
            ) : (
              <div className="text-center text-slate-500 text-sm">
                [Inference Output Matrix Visualizer Ready]
              </div>
            )}
          </div>

          <div className="grid grid-cols-2 gap-4 mt-4">
            <div className="bg-slate-950 p-3 rounded border border-border">
              <span className="text-xs text-slate-400 block">Data Loss (MSE)</span>
              <span className="text-lg font-mono font-bold text-slate-200">{metrics.uLoss}</span>
            </div>
            <div className="bg-slate-950 p-3 rounded border border-border">
              <span className="text-xs text-slate-400 block">PDE Residual Loss</span>
              <span className="text-lg font-mono font-bold text-slate-200">{metrics.pdeLoss}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}