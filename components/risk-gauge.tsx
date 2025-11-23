"use client"

import { useEffect, useRef } from "react"
import * as d3 from "d3"
import type { Alert } from "@/components/pathogen-dashboard"

export function RiskGauge({ alerts }: { alerts: Alert[] }) {
  const svgRef = useRef<SVGSVGElement>(null)

  useEffect(() => {
    if (!svgRef.current || alerts.length === 0) return

    const riskScore = calculateRiskScore(alerts)

    const width = 300
    const height = 180
    const margin = { top: 20, right: 20, bottom: 20, left: 20 }
    const innerRadius = 50
    const outerRadius = 70

    // Clear previous content
    d3.select(svgRef.current).selectAll("*").remove()

    const svg = d3
      .select(svgRef.current)
      .attr("width", width)
      .attr("height", height)
      .attr("viewBox", `0 0 ${width} ${height}`)

    const g = svg.append("g").attr("transform", `translate(${width / 2},${height - margin.bottom})`)

    // Background arc
    const arcBackground = d3
      .arc()
      .innerRadius(innerRadius)
      .outerRadius(outerRadius)
      .startAngle(-Math.PI / 2)
      .endAngle(Math.PI / 2)

    g.append("path")
      .attr("d", arcBackground as any)
      .attr("fill", "#334155")

    // Risk arc
    const arcRisk = d3
      .arc()
      .innerRadius(innerRadius)
      .outerRadius(outerRadius)
      .startAngle(-Math.PI / 2)
      .endAngle(-Math.PI / 2 + Math.PI * (riskScore / 100))

    const color = getRiskColor(riskScore)

    g.append("path")
      .attr("d", arcRisk as any)
      .attr("fill", color)

    // Label
    g.append("text")
      .attr("text-anchor", "middle")
      .attr("dy", "0.35em")
      .attr("font-size", "32px")
      .attr("font-weight", "bold")
      .attr("fill", color)
      .text(Math.round(riskScore))

    // Risk level text
    g.append("text")
      .attr("text-anchor", "middle")
      .attr("dy", "2.5em")
      .attr("font-size", "14px")
      .attr("fill", "#cbd5e1")
      .text(getRiskLevel(riskScore))
  }, [alerts])

  return (
    <svg
      ref={svgRef}
      className="w-full"
      role="img"
      aria-label={`Overall pathogen risk level: ${calculateRiskScore(alerts)} out of 100`}
    />
  )
}

function calculateRiskScore(alerts: Alert[]): number {
  if (alerts.length === 0) return 0

  const severityScores = {
    low: 10,
    medium: 35,
    high: 70,
    critical: 100,
  }

  const avgScore = alerts.reduce((sum, alert) => sum + severityScores[alert.severity], 0) / alerts.length

  return Math.min(100, avgScore)
}

function getRiskColor(score: number): string {
  if (score < 25) return "#10b981" // green
  if (score < 50) return "#f59e0b" // amber
  if (score < 75) return "#ef4444" // red
  return "#7c2d12" // dark red
}

function getRiskLevel(score: number): string {
  if (score < 25) return "Low Risk"
  if (score < 50) return "Moderate Risk"
  if (score < 75) return "High Risk"
  return "Critical"
}
