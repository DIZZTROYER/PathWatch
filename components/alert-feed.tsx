"use client"

import type { Alert } from "@/components/pathogen-dashboard"
import { formatDistanceToNow } from "date-fns"

const SEVERITY_BADGES = {
  low: "bg-green-900 text-green-200",
  medium: "bg-amber-900 text-amber-200",
  high: "bg-red-900 text-red-200",
  critical: "bg-red-950 text-red-100",
}

export function AlertFeed({ alerts }: { alerts: Alert[] }) {
  return (
    <div className="space-y-3" role="region" aria-label="Recent pathogen alerts feed">
      {alerts.length === 0 ? (
        <p className="text-slate-400 text-sm">No alerts yet</p>
      ) : (
        alerts.map((alert) => (
          <div
            key={alert.id}
            className="p-3 bg-slate-700 rounded-lg border border-slate-600 hover:border-slate-500 transition-colors"
            role="article"
            aria-label={`Alert: ${alert.pathogenType} detected in ${alert.location}`}
          >
            <div className="flex items-start justify-between gap-2 mb-2">
              <span className={`text-xs font-semibold px-2 py-1 rounded ${SEVERITY_BADGES[alert.severity]}`}>
                {alert.severity.toUpperCase()}
              </span>
              <span className="text-xs text-slate-400">
                {formatDistanceToNow(new Date(alert.timestamp), { addSuffix: true })}
              </span>
            </div>
            <p className="text-sm font-medium text-white">{alert.pathogenType}</p>
            <p className="text-xs text-slate-400">{alert.location}</p>
          </div>
        ))
      )}
    </div>
  )
}
