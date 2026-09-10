import { useState, useEffect } from 'react'
import type { FieldRequest, FieldResponse, FieldVariable, HealthResponse } from './types/api'
import { checkHealth, fetchFieldPrediction, fetchReferenceField } from './services/api'
import { HeaderBar } from './components/HeaderBar'
import { QueryControls } from './components/QueryControls'
import { FieldVisualization } from './components/FieldVisualization'
import { MetricsFooter } from './components/MetricsFooter'

export function App() {
  const [health, setHealth] = useState<HealthResponse>({ 
    status: 'offline', checkpoint: null, reference_dataset_available: false 
  })
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [params, setParams] = useState<FieldRequest>({
    x_min: -1.0, x_max: 8.0,
    y_min: -2.0, y_max: 2.0,
    t: 0.0, nx: 100, ny: 50,
    compute_vorticity: false,
  })

  const [fieldData, setFieldData] = useState<FieldResponse | null>(null)
  const [refData, setRefData] = useState<FieldResponse | null>(null)
  const [activeVar, setActiveVar] = useState<FieldVariable>('u')
  const [compareMode, setCompareMode] = useState(false)
  
  const [latencyMs, setLatencyMs] = useState<number | null>(null)
  const [l2Error, setL2Error] = useState<number | null>(null)

  useEffect(() => {
    const poll = async () => setHealth(await checkHealth())
    poll()
    const interval = setInterval(poll, 10000)
    return () => clearInterval(interval)
  }, [])

  const handleEvaluate = async () => {
    setIsLoading(true)
    setError(null)
    setL2Error(null)
    
    try {
      const { data, latencyMs: lat } = await fetchFieldPrediction(params)
      setFieldData(data)
      setLatencyMs(lat)

      if (compareMode && health.reference_dataset_available) {
        const ref = await fetchReferenceField(params)
        setRefData(ref)

        // Compute actual L2 error against real ground truth
        let diffSqSum = 0
        let refSqSum = 0
        for (let r = 0; r < data.u.length; r++) {
          for (let c = 0; c < data.u[0].length; c++) {
            diffSqSum += (data.u[r][c] - ref.u[r][c]) ** 2
            refSqSum += ref.u[r][c] ** 2
          }
        }
        setL2Error(Math.sqrt(diffSqSum / (refSqSum || 1.0)))
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to query network endpoint')
    } finally {
      setIsLoading(false)
    }
  }

  // Fetch reference data automatically if user toggles compare mode while data is loaded
  useEffect(() => {
    if (compareMode && fieldData && !refData && health.reference_dataset_available) {
      handleEvaluate()
    }
  }, [compareMode])

  return (
    <div className="flex min-h-screen flex-col bg-[var(--background)] text-[var(--foreground)]">
      <HeaderBar isLive={health.status === 'ok'} />
      <main className="flex-1 p-6">
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
          <div className="lg:col-span-1">
            <QueryControls
              params={params}
              setParams={setParams}
              onEvaluate={handleEvaluate}
              isLoading={isLoading}
              error={error}
            />
          </div>
          <div className="lg:col-span-2">
            <FieldVisualization
              data={fieldData}
              refData={refData}
              params={params}
              isLoading={isLoading}
              activeVar={activeVar}
              setActiveVar={setActiveVar}
              compareMode={compareMode}
              setCompareMode={setCompareMode}
              refAvailable={health.reference_dataset_available}
            />
          </div>
        </div>
        <div className="mt-6">
          <MetricsFooter latencyMs={latencyMs} l2Error={l2Error} healthData={health} />
        </div>
      </main>
    </div>
  )
}
export default App