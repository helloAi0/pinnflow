import React from 'react'

interface HeaderBarProps {
  isLive: boolean
}

export const HeaderBar: React.FC<HeaderBarProps> = ({ isLive }) => {
  return (
    <header className="flex items-center justify-between border-b border-[var(--border)] bg-[var(--surface)] px-6 py-3.5">
      <div className="flex items-center space-x-3">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-[var(--color-primary)] text-xs font-black tracking-widest text-white shadow-sm">
          PF
        </div>
        <div>
          <h1 className="text-base font-bold tracking-tight text-[var(--foreground)]">
            PINNflow <span className="text-xs font-medium text-[var(--muted-foreground)]">| Navier-Stokes Engine</span>
          </h1>
          <p className="text-xs text-[var(--muted-foreground)]">
            Fourier-Constrained Neural Network Diagnostics (2D Unsteady Cylinder Flow)
          </p>
        </div>
      </div>

      <div className="flex items-center space-x-4">
        <span className="inline-flex items-center rounded-md border border-[var(--border)] bg-[var(--slate-950)] px-2.5 py-1 text-xs font-mono font-medium text-[var(--foreground)]">
          Re = 100
        </span>
        <div className="flex items-center space-x-2 rounded-full border border-[var(--border)] bg-[var(--slate-950)] px-3 py-1 text-xs">
          <span className={`h-2 w-2 rounded-full ${isLive ? 'bg-[var(--color-success)] animate-pulse' : 'bg-[var(--color-danger)]'}`} />
          <span className="font-mono text-xs font-medium text-[var(--foreground)]">
            {isLive ? 'API LIVE' : 'OFFLINE'}
          </span>
        </div>
      </div>
    </header>
  )
}