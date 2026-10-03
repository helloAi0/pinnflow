import type { FieldRequest, FieldResponse, HealthResponse, MetadataResponse, PointRequest, PointResponse } from '../types/api'

// Dynamically resolve API URL with smart production fallback
export const getApiUrl = (): string => {
  const envUrl = import.meta.env.VITE_API_URL
  if (envUrl && typeof envUrl === 'string' && envUrl.trim() !== '') {
    return envUrl.replace(/\/+$/, '')
  }
  
  // If no environment variable is provided, check if we are running in the browser
  if (typeof window !== 'undefined' && window.location) {
    const hostname = window.location.hostname
    if (hostname !== 'localhost' && hostname !== '127.0.0.1') {
      return window.location.origin
    }
  }
  
  // Default fallback for local development
  return 'http://localhost:8000'
}

const API_URL = getApiUrl()

/**
 * Standard fetch wrapper with custom timeout and without credentials (enabling wildcard CORS)
 */
async function fetchWithTimeout(url: string, options: RequestInit = {}, timeoutMs = 25000): Promise<Response> {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  try {
    const res = await fetch(url, {
      ...options,
      credentials: 'omit',
      signal: controller.signal
    })
    clearTimeout(timer)
    return res
  } catch (err: unknown) {
    clearTimeout(timer)
    if (err instanceof Error && err.name === 'AbortError') {
      throw new Error('Request timed out. Render backend may be undergoing a cold-start boot or processing high load.')
    }
    throw err
  }
}

/**
 * Helper to parse backend error responses and format actionable messages (e.g. 502 OOM, 504 Gateway Timeout)
 */
async function handleResponseError(res: Response, defaultMessage: string): Promise<never> {
  if (res.status === 502) {
    throw new Error(
      'Backend memory limit exceeded (512MB RAM OOM / HTTP 502 Bad Gateway). The container is restarting. Please reduce the grid resolution (nx / ny) or evaluate a smaller window.'
    )
  }
  if (res.status === 504) {
    throw new Error(
      'Backend request timed out (HTTP 504 Gateway Timeout). Render service may still be waking up. Please retry in a few seconds.'
    )
  }
  if (res.status === 503) {
    const errJson = await res.json().catch(() => null)
    throw new Error(errJson?.detail ?? 'PINNFlow service initializing. Checkpoint calibrating or service not ready.')
  }

  const errJson = await res.json().catch(() => null)
  const errDetail = errJson?.detail ?? `${defaultMessage} (HTTP ${res.status})`
  throw new Error(errDetail)
}

/**
 * Robust health check with exponential backoff to tolerate Render 50s cold-start spin-ups
 */
export async function checkHealth(maxRetries = 3, initialDelayMs = 1500): Promise<HealthResponse> {
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      const res = await fetchWithTimeout(`${API_URL}/api/v1/health`, { method: 'GET' }, 12000)
      if (res.ok) {
        return await res.json()
      }
    } catch {
      // If not the final attempt, wait with exponential backoff
      if (attempt < maxRetries - 1) {
        const delay = initialDelayMs * Math.pow(1.5, attempt)
        await new Promise((resolve) => setTimeout(resolve, delay))
      }
    }
  }

  // Graceful offline state
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

export async function fetchMetadata(): Promise<MetadataResponse | null> {
  try {
    const res = await fetchWithTimeout(`${API_URL}/api/v1/metadata`, { method: 'GET' }, 10000)
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
  const res = await fetchWithTimeout(
    `${API_URL}/api/v1/predict/field`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    },
    35000
  )

  if (!res.ok) {
    await handleResponseError(res, 'Field Prediction Failed')
  }

  const data: FieldResponse = await res.json()
  const latencyMs = performance.now() - start
  return { data, latencyMs }
}

export async function fetchReferenceField(
  payload: FieldRequest
): Promise<FieldResponse> {
  const res = await fetchWithTimeout(
    `${API_URL}/api/v1/reference/field`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    },
    30000
  )

  if (!res.ok) {
    await handleResponseError(res, 'Reference Dataset Unavailable')
  }

  return await res.json()
}

export async function fetchPointPrediction(
  payload: PointRequest
): Promise<PointResponse> {
  const res = await fetchWithTimeout(
    `${API_URL}/api/v1/predict/point`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    },
    15000
  )

  if (!res.ok) {
    await handleResponseError(res, 'Point Prediction Error')
  }

  return await res.json()
}

