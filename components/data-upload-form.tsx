"use client"

import type React from "react"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import type { Alert } from "@/components/pathogen-dashboard"
import axios from "axios"

const PATHOGEN_TYPES = ["COVID-19", "Influenza A", "Influenza B", "RSV", "Mpox", "Poliovirus", "Norovirus"]

export function DataUploadForm({ onNewAlert }: { onNewAlert: (alert: Alert) => void }) {
  const [loading, setLoading] = useState(false)
  const [formData, setFormData] = useState({
    location: "",
    pathogenType: "",
    viralLoad: "",
    detectionDate: "",
    confidence: "",
  })

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target
    setFormData((prev) => ({ ...prev, [name]: value }))
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)

    try {
      const response = await axios.post("/api/pathogen/upload", formData)

      const newAlert: Alert = {
        id: `alert-${Date.now()}`,
        timestamp: new Date().toISOString(),
        severity: determineSeverity(Number.parseFloat(formData.viralLoad)),
        message: `${formData.pathogenType} detected in ${formData.location}`,
        location: formData.location,
        pathogenType: formData.pathogenType,
      }

      onNewAlert(newAlert)

      setFormData({
        location: "",
        pathogenType: "",
        viralLoad: "",
        detectionDate: "",
        confidence: "",
      })
    } catch (error) {
      console.error("Upload failed:", error)
    } finally {
      setLoading(false)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label htmlFor="location" className="block text-sm font-medium text-slate-300 mb-2">
            <span className="sr-only">Location name</span>
            Location
          </label>
          <input
            type="text"
            id="location"
            name="location"
            value={formData.location}
            onChange={handleChange}
            placeholder="Enter detection location"
            required
            aria-label="Detection location"
            className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-amber-500"
          />
        </div>

        <div>
          <label htmlFor="pathogenType" className="block text-sm font-medium text-slate-300 mb-2">
            <span className="sr-only">Select pathogen type</span>
            Pathogen Type
          </label>
          <select
            id="pathogenType"
            name="pathogenType"
            value={formData.pathogenType}
            onChange={handleChange}
            required
            aria-label="Pathogen type"
            className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-amber-500"
          >
            <option value="">Select pathogen</option>
            {PATHOGEN_TYPES.map((pathogen) => (
              <option key={pathogen} value={pathogen}>
                {pathogen}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label htmlFor="viralLoad" className="block text-sm font-medium text-slate-300 mb-2">
            <span className="sr-only">Viral load measurement</span>
            Viral Load (copies/mL)
          </label>
          <input
            type="number"
            id="viralLoad"
            name="viralLoad"
            value={formData.viralLoad}
            onChange={handleChange}
            placeholder="e.g., 1000000"
            required
            aria-label="Viral load in copies per milliliter"
            className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-amber-500"
          />
        </div>

        <div>
          <label htmlFor="confidence" className="block text-sm font-medium text-slate-300 mb-2">
            <span className="sr-only">Confidence level percentage</span>
            Confidence (%)
          </label>
          <input
            type="number"
            id="confidence"
            name="confidence"
            min="0"
            max="100"
            value={formData.confidence}
            onChange={handleChange}
            placeholder="0-100"
            required
            aria-label="Detection confidence percentage"
            className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-amber-500"
          />
        </div>

        <div className="md:col-span-2">
          <label htmlFor="detectionDate" className="block text-sm font-medium text-slate-300 mb-2">
            <span className="sr-only">Detection date and time</span>
            Detection Date
          </label>
          <input
            type="datetime-local"
            id="detectionDate"
            name="detectionDate"
            value={formData.detectionDate}
            onChange={handleChange}
            required
            aria-label="Date and time of detection"
            className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-amber-500"
          />
        </div>
      </div>

      <Button
        type="submit"
        disabled={loading}
        className="w-full bg-amber-500 hover:bg-amber-600 text-slate-900 font-semibold py-2 rounded-md transition-colors"
        aria-label="Submit pathogen detection data"
      >
        {loading ? "Uploading..." : "Upload Detection Data"}
      </Button>
    </form>
  )
}

function determineSeverity(viralLoad: number): "low" | "medium" | "high" | "critical" {
  if (viralLoad < 10000) return "low"
  if (viralLoad < 100000) return "medium"
  if (viralLoad < 1000000) return "high"
  return "critical"
}
