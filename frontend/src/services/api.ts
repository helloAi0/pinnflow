import type { FieldRequest, FieldResponse, HealthResponse, MetadataResponse, PointRequest, PointResponse } from '../types/api'

// Dynamically resolve API URL with smart production fallback
export const getApiUrl = (): string => {
  const envUrl = import.meta.env.VITE_API_URL
  if (envUrl && typeof envUrl === 'string' && envUrl.trim() !== '') {
    return envUrl.replace(/\/+\$/, '')
  }
  
  // If no environment variable is provided, check if we are running in the browser
  if (typeof window !== 'undefined' && window.location) {
    const hostname = window.location.hostname
    
    // If we are on Vercel or any live production domain, infer the Render backend url
    if (hostname !== 'localhost' && hostname !== '127.0.0.1') {
      // Option A: If your Render URL uses a predictable format based on your project name:
      return 'https://onrender.com' 
      
      // Option B: If you prefer to fallback to the current origin (e.g. if using a proxy/same domain):
      // return window.location.origin
    }
  }
  
  // Default fallback for local development
  return 'http://localhost:8000'
}

const API_URL = getApiUrl()

async function fetchWithTimeout(url: string, options: RequestInit = {}, timeoutMs = 15000): Promise<Response> {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  try {
    const res = await fetch(url, { ...options, signal: controller.signal })
    clearTimeout(timer)
    return res
  } catch (err: unknown) {
    clearTimeout(timer)
    if (err instanceof Error && err.name === 'AbortError') {
      throw new Error('Request timed out. If using free-tier Render, the backend service may be spinning up.')
    }
    throw err
  }
}

export async function checkHealth(): Promise<HealthResponse> {
  try {
    const res = await fetchWithTimeout(`${API_URL}/api/v1/health`, { method: 'GET' }, 8000)
    if (!res.ok) throw new Error(`Health check returned status ${res.status}`)
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
    const res = await fetchWithTimeout(`${API_URL}/api/v1/metadata`, { method: 'GET' }, 8000)
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
  const res = await fetchWithTimeout(`${API_URL}/api/v1/predict/field`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  }, 20000)

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
  const res = await fetchWithTimeout(`${API_URL}/api/v1/reference/field`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  }, 20000)

  if (!res.ok) {
    const errJson = await res.json().catch(() => null)
    const errDetail = errJson?.detail ?? `HTTP ${res.status}: Reference Dataset Unavailable`
    throw new Error(errDetail)
  }

  return await res.json()
}

export async function fetchPointPrediction(
  payload: PointRequest
): Promise<PointResponse> {
  const res = await fetchWithTimeout(`${API_URL}/api/v1/predict/point`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  }, 10000)

  if (!res.ok) {
    const errJson = await res.json().catch(() => null)
    const errDetail = errJson?.detail ?? `HTTP ${res.status}: Point Prediction Error`
    throw new Error(errDetail)
  }

  return await res.json()
}
