"use client"

import { Button } from "@/components/ui/button"

const PATHOGEN_TYPES = ["COVID-19", "Influenza A", "Influenza B", "RSV", "Mpox", "Poliovirus", "Norovirus"]

export function FilterPanel({
  selectedPathogen,
  timeRange,
  onPathogenChange,
  onTimeRangeChange,
}: {
  selectedPathogen: string
  timeRange: "24h" | "7d" | "30d"
  onPathogenChange: (pathogen: string) => void
  onTimeRangeChange: (range: "24h" | "7d" | "30d") => void
}) {
  return (
    <div
      className="flex flex-col md:flex-row gap-4 flex-wrap items-center"
      role="search"
      aria-label="Filter pathogen data"
    >
      <div className="flex gap-2 flex-wrap">
        <span className="text-sm font-medium text-slate-400">Pathogen:</span>
        <select
          value={selectedPathogen}
          onChange={(e) => onPathogenChange(e.target.value)}
          aria-label="Filter by pathogen type"
          className="px-3 py-1 bg-slate-700 border border-slate-600 rounded-md text-sm text-white focus:outline-none focus:ring-2 focus:ring-amber-500"
        >
          <option value="">All Pathogens</option>
          {PATHOGEN_TYPES.map((pathogen) => (
            <option key={pathogen} value={pathogen}>
              {pathogen}
            </option>
          ))}
        </select>
      </div>

      <div className="flex gap-2 flex-wrap">
        <span className="text-sm font-medium text-slate-400">Time Range:</span>
        <div className="flex gap-1">
          {(["24h", "7d", "30d"] as const).map((range) => (
            <Button
              key={range}
              onClick={() => onTimeRangeChange(range)}
              variant={timeRange === range ? "default" : "outline"}
              size="sm"
              aria-pressed={timeRange === range}
              aria-label={`Show data from last ${range}`}
              className={
                timeRange === range
                  ? "bg-amber-500 text-slate-900 hover:bg-amber-600"
                  : "bg-slate-700 text-slate-300 hover:bg-slate-600"
              }
            >
              {range}
            </Button>
          ))}
        </div>
      </div>
    </div>
  )
}
