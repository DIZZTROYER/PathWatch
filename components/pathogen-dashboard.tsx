"use client"

import { useState } from "react"
import { Card } from "@/components/ui/card"
import { DataUploadForm } from "./data-upload-form"
import { PathogenMap } from "./pathogen-map"
import { RiskGauge } from "./risk-gauge"
import { AlertFeed } from "./alert-feed"
import { PathogenTimeline } from "./pathogen-timeline"
import { FilterPanel } from "./filter-panel"

export interface Alert {
  id: string
  timestamp: string
  severity: "low" | "medium" | "high" | "critical"
  message: string
  location: string
  pathogenType: string
}

export function PathogenDashboard({ initialAlerts }: { initialAlerts: Alert[] }) {
  const [alerts, setAlerts] = useState<Alert[]>(initialAlerts)
  const [selectedPathogen, setSelectedPathogen] = useState<string>("")
  const [timeRange, setTimeRange] = useState<"24h" | "7d" | "30d">("24h")
  const [mapData, setMapData] = useState<any[]>([])

  const handleNewAlert = (alert: Alert) => {
    setAlerts((prev) => [alert, ...prev].slice(0, 100))
  }

  return (
    <div className="p-4 md:p-8 space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <h1 className="text-3xl md:text-4xl font-bold text-white mb-2">Pathogen Surveillance Dashboard</h1>
          <p className="text-slate-400">
            Real-time monitoring of wastewater pathogen detection and clinical confirmation
          </p>
        </div>
      </div>

      {/* Filters */}
      <FilterPanel
        selectedPathogen={selectedPathogen}
        timeRange={timeRange}
        onPathogenChange={setSelectedPathogen}
        onTimeRangeChange={setTimeRange}
      />

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column */}
        <div className="lg:col-span-2 space-y-6">
          {/* Map */}
          <Card className="bg-slate-800 border-slate-700">
            <div className="p-4 md:p-6">
              <h2 className="text-xl font-semibold text-white mb-4">Geospatial Risk Distribution</h2>
              <PathogenMap alerts={alerts} selectedPathogen={selectedPathogen} timeRange={timeRange} />
            </div>
          </Card>

          {/* Timeline */}
          <Card className="bg-slate-800 border-slate-700">
            <div className="p-4 md:p-6">
              <h2 className="text-xl font-semibold text-white mb-4">Pathogen Spread Timeline</h2>
              <PathogenTimeline alerts={alerts} selectedPathogen={selectedPathogen} />
            </div>
          </Card>
        </div>

        {/* Right Column */}
        <div className="space-y-6">
          {/* Risk Gauge */}
          <Card className="bg-slate-800 border-slate-700">
            <div className="p-4 md:p-6">
              <h2 className="text-lg font-semibold text-white mb-4">Overall Risk Level</h2>
              <RiskGauge alerts={alerts} />
            </div>
          </Card>

          {/* Alert Feed */}
          <Card className="bg-slate-800 border-slate-700">
            <div className="p-4 md:p-6">
              <h2 className="text-lg font-semibold text-white mb-4">Recent Alerts</h2>
              <AlertFeed alerts={alerts.slice(0, 5)} />
            </div>
          </Card>
        </div>
      </div>

      {/* Data Upload */}
      <Card className="bg-slate-800 border-slate-700">
        <div className="p-4 md:p-6">
          <h2 className="text-xl font-semibold text-white mb-4">Upload Detection Data</h2>
          <DataUploadForm onNewAlert={handleNewAlert} />
        </div>
      </Card>
    </div>
  )
}
