import type { FieldRequest, FieldResponse, HealthResponse, MetadataResponse } from '../types/api'

const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

export async function checkHealth(): Promise<HealthResponse> {
  try {
    const res = await fetch(`${API_URL}/api/v1/health`, { method: 'GET' })
    if (!res.ok) throw new Error('Health check non-200')
    return await res.json()
  } catch {
    return {
      status: 'offline',
      trained: false,
      checkpoint_name: null,
      checkpoint_sha: null,
      model_version: '2.0.0',
      training_seed: null,
      git_commit: 'unknown',
      architecture: null,
      reynolds_number: 100.0,
      reference_dataset_available: false
    }
  }
}

export async function fetchMetadata(): Promise<MetadataResponse | null> {
  try {
    const res = await fetch(`${API_URL}/api/v1/metadata`, { method: 'GET' })
    if (!res.ok) return null
    return await res.json()
  } catch {
    return null
  }
}

export async function fetchFieldPrediction(
  payload: FieldRequest
): Promise<{ data: FieldResponse; latencyMs: number }> {
  const start = performance.now()
  const res = await fetch(`${API_URL}/api/v1/predict/field`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })

  if (!res.ok) {
    const errJson = await res.json().catch(() => null)
    const errDetail = errJson?.detail ?? `HTTP ${res.status}: Server Error`
    throw new Error(errDetail)
  }

  const data: FieldResponse = await res.json()
  const latencyMs = performance.now() - start
  return { data, latencyMs }
}

export async function fetchReferenceField(
  payload: FieldRequest
): Promise<FieldResponse> {
  const res = await fetch(`${API_URL}/api/v1/reference/field`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })

  if (!res.ok) {
    const errJson = await res.json().catch(() => null)
    const errDetail = errJson?.detail ?? `HTTP ${res.status}: Reference Dataset Unavailable`
    throw new Error(errDetail)
  }

  return await res.json()
}