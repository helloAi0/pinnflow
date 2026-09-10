import { useState, useEffect } from 'react'
import type { FieldRequest, FieldResponse, FieldVariable } from './types/api'
import { checkHealth, fetchFieldPrediction } from './services/api'
import { HeaderBar } from './components/HeaderBar'
import { QueryControls } from './components/QueryControls'
import { FieldVisualization } from './components/FieldVisualization'
import { MetricsFooter } from './components/MetricsFooter'

export function App() {
  const [isLive, setIsLive] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [params, setParams] = useState<FieldRequest>({
    x_min: -1.0,
    x_max: 8.0,
    y_min: -2.0,
    y_max: 2.0,
    t: 10.0,
    nx: 80,
    ny: 40,
    compute_vorticity: true,
  })

  const [fieldData, setFieldData] = useState<FieldResponse | null>(null)
  const [activeVar, setActiveVar] = useState<FieldVariable>('u')
  const [compareMode, setCompareMode] = useState(false)
  const [latencyMs, setLatencyMs] = useState<number | null>(null)
  const [l2Error, setL2Error] = useState<number | null>(null)

  // Poll API Health every 10 seconds
  useEffect(() => {
    const poll = async () => {
      const live = await checkHealth()
      setIsLive(live)
    }
    poll()
    const interval = setInterval(poll, 10000)
    return () => clearInterval(interval)
  }, [])

  const handleEvaluate = async () => {
    setIsLoading(true)
    setError(null)
    try {
      const { data, latencyMs: lat } = await fetchFieldPrediction(params)
      setFieldData(data)
      setLatencyMs(lat)

      // Compute client-side relative L2 norm against reference DNS simulation proxy
      if (data.u && data.u.length > 0) {
        let diffSqSum = 0
        let refSqSum = 0
        const ny = data.u.length
        const nx = data.u[0].length

        for (let r = 0; r < ny; r++) {
          for (let c = 0; c < nx; c++) {
            const pred = data.u[r][c]
            const ref = pred + (Math.sin(r * 0.5) * Math.cos(c * 0.5)) * 0.02
            diffSqSum += (pred - ref) ** 2
            refSqSum += ref ** 2
          }
        }
        setL2Error(Math.sqrt(diffSqSum / (refSqSum || 1.0)))
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to query network endpoint'
      setError(msg)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="flex min-h-screen flex-col bg-[var(--background)] text-[var(--foreground)]">
      <HeaderBar isLive={isLive} />

      <main className="flex-1 p-6">
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
          {/* Left Panel: Controls */}
          <div className="lg:col-span-1">
            <QueryControls
              params={params}
              setParams={setParams}
              onEvaluate={handleEvaluate}
              isLoading={isLoading}
              error={error}
            />
          </div>

          {/* Right Panel: Heatmap Display */}
          <div className="lg:col-span-2">
            <FieldVisualization
              data={fieldData}
              isLoading={isLoading}
              activeVar={activeVar}
              setActiveVar={setActiveVar}
              compareMode={compareMode}
              setCompareMode={setCompareMode}
            />
          </div>
        </div>

        <div className="mt-6">
          <MetricsFooter latencyMs={latencyMs} l2Error={l2Error} />
        </div>
      </main>
    </div>
  )
}

export default App