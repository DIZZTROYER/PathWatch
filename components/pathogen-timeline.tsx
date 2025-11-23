"use client"

import { useEffect, useRef } from "react"
import * as d3 from "d3"
import type { Alert } from "@/components/pathogen-dashboard"

interface TimelineEvent {
  stage: string
  date: Date
  count: number
}

export function PathogenTimeline({
  alerts,
  selectedPathogen,
}: {
  alerts: Alert[]
  selectedPathogen: string
}) {
  const svgRef = useRef<SVGSVGElement>(null)

  useEffect(() => {
    if (!svgRef.current) return

    const filteredAlerts = alerts.filter((alert) => {
      if (selectedPathogen && alert.pathogenType !== selectedPathogen) return false
      return true
    })

    if (filteredAlerts.length === 0) {
      d3.select(svgRef.current).selectAll("*").remove()
      return
    }

    // Generate timeline data
    const timelineData: TimelineEvent[] = [
      { stage: "Wastewater\nDetection", date: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000), count: 12 },
      { stage: "Confirmation\nTesting", date: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000), count: 8 },
      { stage: "Clinical\nCases", date: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000), count: 3 },
    ]

    const width = 600
    const height = 250
    const margin = { top: 40, right: 40, bottom: 40, left: 40 }

    d3.select(svgRef.current).selectAll("*").remove()

    const svg = d3
      .select(svgRef.current)
      .attr("width", width)
      .attr("height", height)
      .attr("viewBox", `0 0 ${width} ${height}`)

    const xScale = d3
      .scaleLinear()
      .domain([0, timelineData.length - 1])
      .range([margin.left, width - margin.right])

    const yScale = d3
      .scaleLinear()
      .domain([0, Math.max(...timelineData.map((d) => d.count)) + 2])
      .range([height - margin.bottom, margin.top])

    // Draw connecting line
    svg
      .append("line")
      .attr("x1", xScale(0))
      .attr("y1", height / 2)
      .attr("x2", xScale(timelineData.length - 1))
      .attr("y2", height / 2)
      .attr("stroke", "#cbd5e1")
      .attr("stroke-width", 2)

    // Draw circles and bars
    timelineData.forEach((d, i) => {
      const x = xScale(i)
      const y = yScale(d.count)

      // Bar
      svg
        .append("rect")
        .attr("x", x - 15)
        .attr("y", y)
        .attr("width", 30)
        .attr("height", height - margin.bottom - y)
        .attr("fill", "#f59e0b")
        .attr("opacity", 0.6)

      // Circle
      svg
        .append("circle")
        .attr("cx", x)
        .attr("cy", height / 2)
        .attr("r", 8)
        .attr("fill", "#fbbf24")
        .attr("stroke", "#fff")
        .attr("stroke-width", 2)

      // Label
      svg
        .append("text")
        .attr("x", x)
        .attr("y", height - 5)
        .attr("text-anchor", "middle")
        .attr("font-size", "12px")
        .attr("fill", "#cbd5e1")
        .text(d.stage)

      // Count
      svg
        .append("text")
        .attr("x", x)
        .attr("y", y - 10)
        .attr("text-anchor", "middle")
        .attr("font-size", "14px")
        .attr("font-weight", "bold")
        .attr("fill", "#fbbf24")
        .text(d.count)
    })
  }, [alerts, selectedPathogen])

  return (
    <div
      className="w-full overflow-x-auto"
      role="region"
      aria-label="Pathogen spread timeline from detection to clinical confirmation"
    >
      <svg ref={svgRef} className="min-w-full h-64" />
    </div>
  )
}
