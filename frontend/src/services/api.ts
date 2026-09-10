import type { FieldRequest, FieldResponse, HealthResponse } from '../types/api'

const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_URL}/health`, { method: 'GET' })
    if (!res.ok) return false
    const data: HealthResponse = await res.json().catch(() => ({ status: 'ok' }))
    return res.status === 200 || data.status === 'ok'
  } catch {
    return false
  }
}

export async function fetchFieldPrediction(
  payload: FieldRequest
): Promise<{ data: FieldResponse; latencyMs: number }> {
  const start = performance.now()
  const res = await fetch(`${API_URL}/predict/field`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })

  if (!res.ok) {
    const errText = await res.text().catch(() => 'Unknown server error')
    throw new Error(`HTTP ${res.status}: ${errText}`)
  }

  const data: FieldResponse = await res.json()
  const latencyMs = performance.now() - start
  return { data, latencyMs }
}