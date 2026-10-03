import type { FieldRequest, FieldResponse, HealthResponse, MetadataResponse, PointRequest, PointResponse } from '../types/api'

// Dynamically resolve API URL with smart production fallback
export const getApiUrl = (): string => {
  const envUrl = import.meta.env.VITE_API_URL
  if (envUrl && typeof envUrl === 'string' && envUrl.trim() !== '') {
    return envUrl.replace(/\/+$/, '')
  }
  
  // When running on non-localhost (e.g. Vercel deployment), point directly to Render production backend
  if (typeof window !== 'undefined' && window.location) {
    const hostname = window.location.hostname
    if (hostname !== 'localhost' && hostname !== '127.0.0.1') {
      return 'https://pinnflow.onrender.com'
    }
  }
  
  // Default fallback for local development
  return 'http://localhost:8000'
}

const API_URL = getApiUrl()

/**
 * Robust fetch wrapper with 60,000ms timeout and exponential backoff retry logic.
 */
export async function fetchWithTimeoutAndRetry(
  url: string,
  options: RequestInit = {},
  maxRetries = 2,
  timeoutMs = 60000
): Promise<Response> {
  let lastError: unknown = null

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), timeoutMs)

    try {
      const res = await fetch(url, {
        ...options,
        credentials: 'omit',
        signal: controller.signal
      })
      clearTimeout(timer)

      // If response is successful or client-side 4xx (not 502/504), return immediately
      if (res.ok || (res.status >= 400 && res.status < 500 && res.status !== 408)) {
        return res
      }

      // If server returned 502 / 504 / 503, retry on remaining attempts
      if (attempt < maxRetries && (res.status === 502 || res.status === 504 || res.status === 503)) {
        const delay = 2000 * Math.pow(1.5, attempt)
        await new Promise((r) => setTimeout(r, delay))
        continue
      }

      return res
    } catch (err: unknown) {
      clearTimeout(timer)
      lastError = err

      if (attempt < maxRetries) {
        const delay = 2000 * Math.pow(1.5, attempt)
        await new Promise((r) => setTimeout(r, delay))
      }
    }
  }

  if (lastError instanceof Error && lastError.name === 'AbortError') {
    throw new Error(
      'Request timed out (60s limit). If using free-tier Render, the backend server may still be spinning up. Please try again in 10-20 seconds.'
    )
  }

  throw lastError || new Error('Network connection failed.')
}

/**
 * Helper to parse backend error responses and format actionable messages
 */
async function handleResponseError(res: Response, defaultMessage: string): Promise<never> {
  if (res.status === 502) {
    throw new Error(
      'Backend memory limit exceeded (512MB RAM OOM / HTTP 502 Bad Gateway). The container is restarting. Please reduce the grid resolution (nx / ny) or evaluate a smaller window.'
    )
  }
  if (res.status === 504) {
    throw new Error(
      'Backend request timed out (HTTP 504 Gateway Timeout). Render service may still be waking up. Please retry in a few moments.'
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
 * Lightweight ping to trigger cold-start wakeup on Render
 */
export async function pingWakeup(): Promise<boolean> {
  try {
    const res = await fetchWithTimeoutAndRetry(`${API_URL}/api/v1/wakeup`, { method: 'GET' }, 1, 15000)
    return res.ok
  } catch {
    return false
  }
}

/**
 * Robust health check with exponential backoff to tolerate Render 50s cold-start spin-ups
 */
export async function checkHealth(maxRetries = 3, initialDelayMs = 1500): Promise<HealthResponse> {
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      const res = await fetchWithTimeoutAndRetry(`${API_URL}/api/v1/health`, { method: 'GET' }, 0, 20000)
      if (res.ok) {
        return await res.json()
      }
    } catch {
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
    const res = await fetchWithTimeoutAndRetry(`${API_URL}/api/v1/metadata`, { method: 'GET' }, 1, 15000)
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
  const res = await fetchWithTimeoutAndRetry(
    `${API_URL}/api/v1/predict/field`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    },
    2,
    60000
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
  const res = await fetchWithTimeoutAndRetry(
    `${API_URL}/api/v1/reference/field`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    },
    2,
    60000
  )

  if (!res.ok) {
    await handleResponseError(res, 'Reference Dataset Unavailable')
  }

  return await res.json()
}

export async function fetchPointPrediction(
  payload: PointRequest
): Promise<PointResponse> {
  const res = await fetchWithTimeoutAndRetry(
    `${API_URL}/api/v1/predict/point`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    },
    1,
    20000
  )

  if (!res.ok) {
    await handleResponseError(res, 'Point Prediction Error')
  }

  return await res.json()
}


