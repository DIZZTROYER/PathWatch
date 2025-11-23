"use client"

import { useMemo } from "react"
import type { Alert } from "@/components/pathogen-dashboard"

const SEVERITY_COLORS = {
  low: "#10b981",
  medium: "#f59e0b",
  high: "#ef4444",
  critical: "#7c2d12",
}

// US state center points for demo (latitude, longitude)
const US_STATES = {
  California: { lat: 36.7783, lng: -119.4179, x: 80, y: 300 },
  Texas: { lat: 31.9686, lng: -99.9018, x: 420, y: 380 },
  Florida: { lat: 27.6648, lng: -81.5158, x: 580, y: 420 },
  "New York": { lat: 43.2994, lng: -74.2179, x: 600, y: 200 },
  Illinois: { lat: 40.3495, lng: -88.9861, x: 520, y: 260 },
  Georgia: { lat: 33.7537, lng: -83.4243, x: 560, y: 360 },
  Pennsylvania: { lat: 40.8258, lng: -77.0369, x: 610, y: 240 },
  Ohio: { lat: 40.3888, lng: -82.7649, x: 550, y: 280 },
}

export function PathogenMap({
  alerts,
  selectedPathogen,
}: {
  alerts: Alert[]
  selectedPathogen: string
}) {
  const markers = useMemo(() => {
    const stateNames = Object.keys(US_STATES)
    const filteredAlerts = alerts.filter((alert) => {
      if (selectedPathogen && alert.pathogenType !== selectedPathogen) return false
      return true
    })

    return filteredAlerts.map((alert) => {
      const state = stateNames[Math.floor(Math.random() * stateNames.length)]
      const coords = US_STATES[state as keyof typeof US_STATES]
      const radius = alert.severity === "critical" ? 10 : alert.severity === "high" ? 8 : 6

      return {
        ...alert,
        state,
        x: coords.x + (Math.random() - 0.5) * 30,
        y: coords.y + (Math.random() - 0.5) * 30,
        radius,
      }
    })
  }, [alerts, selectedPathogen])

  return (
    <div
      role="region"
      aria-label="Pathogen distribution map showing detected locations across United States"
      className="w-full bg-slate-900 rounded-lg border border-slate-700 overflow-hidden"
    >
      <svg
        viewBox="0 0 700 500"
        className="w-full h-96"
        xmlns="http://www.w3.org/2000/svg"
        role="img"
        aria-label="US map with pathogen detection markers"
      >
        {/* Map background */}
        <rect width="700" height="500" fill="#1e293b" />

        {/* Simplified US state regions as guides */}
        <g fill="none" stroke="#334155" strokeWidth="1" opacity="0.3">
          {/* West Coast outline */}
          <path d="M 80 200 L 120 300 L 100 400" />
          {/* Mountain West outline */}
          <path d="M 150 150 L 250 200 L 250 400" />
          {/* Central outline */}
          <path d="M 300 100 L 450 150 L 500 400" />
          {/* Eastern outline */}
          <path d="M 500 100 L 650 120 L 650 450" />
        </g>

        {/* Grid lines for reference */}
        <g stroke="#475569" strokeWidth="0.5" opacity="0.2">
          {Array.from({ length: 7 }).map((_, i) => (
            <line key={`vline-${i}`} x1={(i * 700) / 6} y1="0" x2={(i * 700) / 6} y2="500" />
          ))}
          {Array.from({ length: 5 }).map((_, i) => (
            <line key={`hline-${i}`} x1="0" y1={(i * 500) / 4} x2="700" y2={(i * 500) / 4} />
          ))}
        </g>

        {/* State location markers (reference points) */}
        {Object.entries(US_STATES).map(([state, coords]) => (
          <circle key={`state-${state}`} cx={coords.x} cy={coords.y} r="3" fill="#64748b" opacity="0.5" />
        ))}

        {/* Pathogen detection markers */}
        {markers.map((marker, idx) => (
          <g key={marker.id}>
            <circle
              cx={marker.x}
              cy={marker.y}
              r={marker.radius + 2}
              fill={SEVERITY_COLORS[marker.severity]}
              opacity="0.3"
            />
            <circle
              cx={marker.x}
              cy={marker.y}
              r={marker.radius}
              fill={SEVERITY_COLORS[marker.severity]}
              stroke="#fff"
              strokeWidth="2"
              style={{ cursor: "pointer" }}
            >
              <title>{`${marker.pathogenType} - ${marker.severity.toUpperCase()} - ${marker.state}`}</title>
            </circle>

            {/* Animated pulse effect for critical alerts */}
            {marker.severity === "critical" && (
              <circle
                cx={marker.x}
                cy={marker.y}
                r={marker.radius}
                fill="none"
                stroke={SEVERITY_COLORS[marker.severity]}
                strokeWidth="1"
                opacity="0.5"
              >
                <animate
                  attributeName="r"
                  from={marker.radius}
                  to={marker.radius + 6}
                  dur="2s"
                  repeatCount="indefinite"
                />
                <animate attributeName="opacity" from="0.8" to="0" dur="2s" repeatCount="indefinite" />
              </circle>
            )}
          </g>
        ))}
      </svg>

      {/* Legend */}
      <div className="p-4 bg-slate-900 border-t border-slate-700">
        <div className="flex flex-wrap gap-4 text-xs">
          {Object.entries(SEVERITY_COLORS).map(([severity, color]) => (
            <div key={severity} className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full" style={{ backgroundColor: color }} />
              <span className="text-slate-400 capitalize">{severity}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
