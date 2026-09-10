import type { FieldRequest, FieldResponse, HealthResponse } from '../types/api'

const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

export async function checkHealth(): Promise<HealthResponse> {
  try {
    const res = await fetch(`${API_URL}/health`, { method: 'GET' })
    if (!res.ok) throw new Error('Network error')
    return await res.json()
  } catch {
    return { status: 'offline', checkpoint: null, reference_dataset_available: false }
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

export async function fetchReferenceField(
  payload: FieldRequest
): Promise<FieldResponse> {
  const res = await fetch(`${API_URL}/reference/field`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })

  if (!res.ok) {
    const errText = await res.text().catch(() => 'Unknown server error')
    throw new Error(`Reference HTTP ${res.status}: ${errText}`)
  }

  return await res.json()
}