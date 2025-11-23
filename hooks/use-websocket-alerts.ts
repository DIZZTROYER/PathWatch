"use client"

import { useState, useEffect } from "react"
import type { Alert } from "@/components/pathogen-dashboard"

export function useWebSocketAlerts(): Alert[] {
  const [alerts, setAlerts] = useState<Alert[]>([
    {
      id: "demo-1",
      timestamp: new Date(Date.now() - 2 * 60 * 1000).toISOString(),
      severity: "high",
      message: "COVID-19 detected in wastewater treatment facility",
      location: "New York, NY",
      pathogenType: "COVID-19",
    },
    {
      id: "demo-2",
      timestamp: new Date(Date.now() - 15 * 60 * 1000).toISOString(),
      severity: "medium",
      message: "Influenza A confirmed in clinical samples",
      location: "Los Angeles, CA",
      pathogenType: "Influenza A",
    },
    {
      id: "demo-3",
      timestamp: new Date(Date.now() - 45 * 60 * 1000).toISOString(),
      severity: "low",
      message: "RSV traces detected in wastewater",
      location: "Chicago, IL",
      pathogenType: "RSV",
    },
  ])

  useEffect(() => {
    // Simulate WebSocket connection
    const interval = setInterval(() => {
      // Randomly add new alerts for demo purposes
      if (Math.random() > 0.7) {
        const pathogenTypes = ["COVID-19", "Influenza A", "RSV", "Mpox", "Norovirus"]
        const locations = ["Seattle, WA", "Boston, MA", "Miami, FL", "Denver, CO", "Atlanta, GA"]
        const severities: ("low" | "medium" | "high" | "critical")[] = ["low", "medium", "high"]

        const newAlert: Alert = {
          id: `alert-${Date.now()}`,
          timestamp: new Date().toISOString(),
          severity: severities[Math.floor(Math.random() * severities.length)],
          message: `${pathogenTypes[Math.floor(Math.random() * pathogenTypes.length)]} detected`,
          location: locations[Math.floor(Math.random() * locations.length)],
          pathogenType: pathogenTypes[Math.floor(Math.random() * pathogenTypes.length)],
        }

        setAlerts((prev) => [newAlert, ...prev].slice(0, 100))
      }
    }, 30000)

    return () => clearInterval(interval)
  }, [])

  return alerts
}
