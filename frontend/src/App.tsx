import React, { useState, useEffect } from 'react'
import type { FieldRequest, FieldResponse, FieldVariable, HealthResponse, VisualizationMode, ActiveTab } from './types/api'
import { checkHealth, fetchFieldPrediction, fetchReferenceField } from './services/api'
import { HeaderBar } from './components/HeaderBar'
import { QueryControls } from './components/QueryControls'
import { FieldVisualization } from './components/FieldVisualization'
import { MetricsFooter } from './components/MetricsFooter'
import { BenchmarkTab } from './components/BenchmarkTab'
import { ReproducibilityTab } from './components/ReproducibilityTab'
import { DiagnosticsTab } from './components/DiagnosticsTab'
import { Activity, Layers, Award, ShieldCheck, BarChart2 } from 'lucide-react'

export function App() {
  const [health, setHealth] = useState<HealthResponse>({
    status: 'checking',
    trained: false,
    checkpoint_name: null,
    checkpoint_sha: null,
    model_version: '2.0.0',
    training_seed: null,
    git_commit: 'unknown',
    architecture: 'hard_constrained_pinn',
    reynolds_number: 100.0,
    reference_dataset_available: false,
  })

  const [activeTab, setActiveTab] = useState<ActiveTab>('field')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Experiment parameters
  const [params, setParams] = useState<FieldRequest>({
    x_min: 1.0,
    x_max: 8.0,
    y_min: -2.0,
    y_max: 2.0,
    t: 10.0,
    nx: 100,
    ny: 50,
    compute_vorticity: true,
  })

  const [selectedModel, setSelectedModel] = useState<string>('hard_constrained_pinn')
  const [selectedBudget, setSelectedBudget] = useState<number>(5000)
  const [selectedNoise, setSelectedNoise] = useState<number>(0.0)
  const [selectedSplit, setSelectedSplit] = useState<string>('random')
  const [selectedSeed, setSelectedSeed] = useState<number>(42)

  const [fieldData, setFieldData] = useState<FieldResponse | null>(null)
  const [refData, setRefData] = useState<FieldResponse | null>(null)
  const [activeVar, setActiveVar] = useState<FieldVariable>('velocity_magnitude')
  const [visMode, setVisMode] = useState<VisualizationMode>('prediction')
  const [compareMode, setCompareMode] = useState(true)

  const [latencyMs, setLatencyMs] = useState<number | null>(null)
  const [l2Error, setL2Error] = useState<number | null>(null)

  useEffect(() => {
    const poll = async () => setHealth(await checkHealth())
    poll()
    const interval = setInterval(poll, 15000)
    return () => clearInterval(interval)
  }, [])

  const handleEvaluate = React.useCallback(async () => {
    setIsLoading(true)
    setError(null)
    setL2Error(null)

    try {
      const { data, latencyMs: lat } = await fetchFieldPrediction(params)
      setFieldData(data)
      setLatencyMs(lat)

      if (compareMode && health.reference_dataset_available) {
        try {
          const ref = await fetchReferenceField(params)
          setRefData(ref)

          // Exact mathematical L2 error computation against authentic DNS data
          let diffSqSum = 0
          let refSqSum = 0
          for (let r = 0; r < data.u.length; r++) {
            for (let c = 0; c < data.u[0].length; c++) {
              const uDiff = data.u[r][c] - ref.u[r][c]
              const vDiff = data.v[r][c] - ref.v[r][c]
              diffSqSum += uDiff * uDiff + vDiff * vDiff
              refSqSum += ref.u[r][c] * ref.u[r][c] + ref.v[r][c] * ref.v[r][c]
            }
          }
          const computedL2 = Math.sqrt(diffSqSum / (refSqSum || 1.0))
          setL2Error(computedL2)
        } catch {
          // Reference DNS fetch failed gracefully
        }
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Evaluation query failed')
    } finally {
      setIsLoading(false)
    }
  }, [params, compareMode, health.reference_dataset_available])

  // Auto-trigger evaluation on initial page load
  useEffect(() => {
    handleEvaluate()
  }, [handleEvaluate])

  const handleExportFieldCSV = () => {
    if (!fieldData) return
    let csvContent = 'data:text/csv;charset=utf-8,x,y,t,u,v,p,velocity_magnitude\n'
    for (let r = 0; r < fieldData.u.length; r++) {
      for (let c = 0; c < fieldData.u[0].length; c++) {
        const xVal = fieldData.x[r][c]
        const yVal = fieldData.y[r][c]
        const uVal = fieldData.u[r][c]
        const vVal = fieldData.v[r][c]
        const pVal = fieldData.p[r][c]
        const velMag = fieldData.velocity_magnitude[r][c]
        csvContent += `${xVal},${yVal},${params.t},${uVal},${vVal},${pVal},${velMag}\n`
      }
    }
    const encodedUri = encodeURI(csvContent)
    const link = document.createElement('a')
    link.setAttribute('href', encodedUri)
    link.setAttribute('download', `pinnflow_field_t${params.t.toFixed(1)}.csv`)
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  const handleExportMetricsJSON = () => {
    if (!fieldData) return
    const exportBundle = {
      timestamp: new Date().toISOString(),
      model: selectedModel,
      reynolds_number: health.reynolds_number,
      seed: selectedSeed,
      budget: selectedBudget,
      noise: selectedNoise,
      split: selectedSplit,
      time_snapshot: params.t,
      grid_resolution: `${params.nx}x${params.ny}`,
      relative_l2_error: l2Error,
      pde_residual_loss: fieldData.metrics.pde_loss,
      latency_ms: latencyMs,
      checkpoint_sha: health.checkpoint_sha,
      git_commit: health.git_commit
    }
    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(exportBundle, null, 2))
    const link = document.createElement('a')
    link.setAttribute('href', dataStr)
    link.setAttribute('download', `pinnflow_metrics_t${params.t.toFixed(1)}.json`)
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  return (
    <div className="flex min-h-screen flex-col bg-slate-950 text-slate-100 antialiased selection:bg-cyan-500 selection:text-black">
      <HeaderBar health={health} />

      {/* Navigation Tab Bar */}
      <div className="border-b border-slate-800 bg-slate-900/40 px-6 backdrop-blur-sm">
        <nav className="flex space-x-6 font-mono text-xs">
          <button
            onClick={() => { setActiveTab('field'); setVisMode('prediction') }}
            className={`flex items-center space-x-2 py-3 border-b-2 font-medium transition ${
              activeTab === 'field'
                ? 'border-cyan-400 text-cyan-400'
                : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            <Activity className="h-4 w-4" />
            <span>Field View</span>
          </button>

          <button
            onClick={() => { setActiveTab('error_analysis'); setVisMode('absolute_error') }}
            className={`flex items-center space-x-2 py-3 border-b-2 font-medium transition ${
              activeTab === 'error_analysis'
                ? 'border-cyan-400 text-cyan-400'
                : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            <BarChart2 className="h-4 w-4" />
            <span>Error Analysis</span>
          </button>

          <button
            onClick={() => setActiveTab('diagnostics')}
            className={`flex items-center space-x-2 py-3 border-b-2 font-medium transition ${
              activeTab === 'diagnostics'
                ? 'border-cyan-400 text-cyan-400'
                : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            <Layers className="h-4 w-4" />
            <span>Physics Diagnostics</span>
          </button>

          <button
            onClick={() => setActiveTab('benchmark')}
            className={`flex items-center space-x-2 py-3 border-b-2 font-medium transition ${
              activeTab === 'benchmark'
                ? 'border-cyan-400 text-cyan-400'
                : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            <Award className="h-4 w-4" />
            <span>Benchmark Matrix</span>
          </button>

          <button
            onClick={() => setActiveTab('reproducibility')}
            className={`flex items-center space-x-2 py-3 border-b-2 font-medium transition ${
              activeTab === 'reproducibility'
                ? 'border-cyan-400 text-cyan-400'
                : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            <ShieldCheck className="h-4 w-4" />
            <span>Reproducibility Manifest</span>
          </button>
        </nav>
      </div>

      {/* Main Scientific Instrument Workspace */}
      <main className="flex-1 p-6 space-y-6">
        {activeTab === 'field' || activeTab === 'error_analysis' ? (
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
            <div className="lg:col-span-1">
              <QueryControls
                params={params}
                setParams={setParams}
                selectedModel={selectedModel}
                setSelectedModel={setSelectedModel}
                selectedBudget={selectedBudget}
                setSelectedBudget={setSelectedBudget}
                selectedNoise={selectedNoise}
                setSelectedNoise={setSelectedNoise}
                selectedSplit={selectedSplit}
                setSelectedSplit={setSelectedSplit}
                selectedSeed={selectedSeed}
                setSelectedSeed={setSelectedSeed}
                onEvaluate={handleEvaluate}
                isLoading={isLoading}
                error={error}
                compareMode={compareMode}
                setCompareMode={setCompareMode}
                onExportField={handleExportFieldCSV}
                onExportMetrics={handleExportMetricsJSON}
              />
            </div>
            <div className="lg:col-span-2">
              <FieldVisualization
                data={fieldData}
                refData={refData}
                activeVar={activeVar}
                setActiveVar={setActiveVar}
                visMode={visMode}
                setVisMode={setVisMode}
                isLoading={isLoading}
                compareMode={compareMode}
              />
            </div>
          </div>
        ) : activeTab === 'diagnostics' ? (
          <DiagnosticsTab data={fieldData} />
        ) : activeTab === 'benchmark' ? (
          <BenchmarkTab />
        ) : (
          <ReproducibilityTab
            health={health}
            onExportField={handleExportFieldCSV}
            onExportMetrics={handleExportMetricsJSON}
          />
        )}

        {/* Global Scientific Metrics Footer */}
        <MetricsFooter
          data={fieldData}
          refData={refData}
          latencyMs={latencyMs}
          l2Error={l2Error}
          health={health}
        />
      </main>
    </div>
  )
}

export default App