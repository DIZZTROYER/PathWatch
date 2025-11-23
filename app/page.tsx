"use client"
import { PathogenDashboard } from "@/components/pathogen-dashboard"
import { useWebSocketAlerts } from "@/hooks/use-websocket-alerts"

export default function Home() {
  const alerts = useWebSocketAlerts()

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-900 to-slate-800">
      <PathogenDashboard initialAlerts={alerts} />
    </main>
  )
}
