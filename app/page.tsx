"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { FolderOpen, FileText, Database, CheckCircle, Download, Filter, ArrowRight, ArrowLeft } from "lucide-react"
import FolderSelection from "@/components/folder-selection"
import FileProcessing from "@/components/file-processing"
import FileSelectionForIngestion from "@/components/file-selection-ingestion"
import DatabaseConfiguration from "@/components/database-configuration"
import IngestionProcess from "@/components/ingestion-process"
import SummaryView from "@/components/summary-view"

export type FileData = {
  id: string
  name: string
  path: string
  size: number
  type: string
  selected: boolean
  processed: boolean
  file?: File // Add the actual File object
  qualityMetrics?: {
    parseAccuracy: number
    completeness: number
    formatConsistency: number
  }
  classification?: "Structured" | "Semi-Structured" | "Unstructured"
  ingestionStatus?: "pending" | "success" | "failed"
  ingestionDetails?: any
}

export type DatabaseConfig = {
  structured: {
    type: "postgresql" | "mysql"
    host: string
    port: number
    database: string
    username: string
    password: string
  }
  unstructured: {
    type: "milvus" | "qdrant"
    host: string
    port: number
    collection: string
    apiKey?: string
  }
}

const steps = [
  { id: 1, title: "Folder Selection", icon: FolderOpen },
  { id: 2, title: "File Processing", icon: FileText },
  { id: 3, title: "File Selection", icon: Filter },
  { id: 4, title: "Database Config", icon: Database },
  { id: 5, title: "Ingestion", icon: ArrowRight },
  { id: 6, title: "Summary", icon: CheckCircle },
]

export default function DataLoaderAutomation() {
  const [currentStep, setCurrentStep] = useState(1)
  const [files, setFiles] = useState<FileData[]>([])
  const [databaseConfig, setDatabaseConfig] = useState<DatabaseConfig>({
    structured: {
      type: "postgresql",
      host: "localhost",
      port: 5432,
      database: "dataloader",
      username: "postgres",
      password: "",
    },
    unstructured: {
      type: "milvus",
      host: "localhost",
      port: 19530,
      collection: "documents",
    },
  })
  const [processingProgress, setProcessingProgress] = useState(0)
  const [ingestionProgress, setIngestionProgress] = useState(0)

  const nextStep = () => {
    if (currentStep < 6) {
      setCurrentStep(currentStep + 1)
    }
  }

  const prevStep = () => {
    if (currentStep > 1) {
      setCurrentStep(currentStep - 1)
    }
  }

  const canProceed = () => {
    switch (currentStep) {
      case 1:
        return files.length > 0
      case 2:
        return files.filter((f) => f.selected).every((f) => f.processed)
      case 3:
        return files.some((f) => f.selected)
      case 4:
        return true // Database config is optional
      case 5:
        return files.filter((f) => f.selected).every((f) => f.ingestionStatus !== "pending")
      default:
        return true
    }
  }

  // Add a new useEffect to automatically proceed to step 6 after step 5 completion
  const autoAdvanceToSummary = () => {
    if (currentStep === 5) {
      const selectedFiles = files.filter((f) => f.selected)
      const allIngestionComplete =
        selectedFiles.length > 0 &&
        selectedFiles.every((f) => f.ingestionStatus === "success" || f.ingestionStatus === "failed")

      if (allIngestionComplete) {
        // Add a small delay for better UX
        setTimeout(() => {
          setCurrentStep(6)
        }, 1500)
      }
    }
  }

  // Call this function whenever files state changes
  useEffect(() => {
    autoAdvanceToSummary()
  }, [files, currentStep])

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-6">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">Data Loader Automation</h1>
          <p className="text-lg text-gray-600">Automated file processing and database ingestion pipeline</p>
        </div>

        {/* Progress Steps */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            {steps.map((step, index) => {
              const Icon = step.icon
              const isActive = currentStep === step.id
              const isCompleted = currentStep > step.id

              return (
                <div key={step.id} className="flex items-center">
                  <div
                    className={`
                    flex items-center justify-center w-12 h-12 rounded-full border-2 transition-all
                    ${
                      isActive
                        ? "bg-blue-600 border-blue-600 text-white"
                        : isCompleted
                          ? "bg-green-600 border-green-600 text-white"
                          : "bg-white border-gray-300 text-gray-400"
                    }
                  `}
                  >
                    <Icon className="w-5 h-5" />
                  </div>
                  <div className="ml-3 hidden sm:block">
                    <p
                      className={`text-sm font-medium ${
                        isActive ? "text-blue-600" : isCompleted ? "text-green-600" : "text-gray-500"
                      }`}
                    >
                      Step {step.id}
                    </p>
                    <p
                      className={`text-xs ${
                        isActive ? "text-blue-600" : isCompleted ? "text-green-600" : "text-gray-500"
                      }`}
                    >
                      {step.title}
                    </p>
                  </div>
                  {index < steps.length - 1 && (
                    <div
                      className={`
                      w-8 h-0.5 mx-4 transition-all
                      ${isCompleted ? "bg-green-600" : "bg-gray-300"}
                    `}
                    />
                  )}
                </div>
              )
            })}
          </div>
        </div>

        {/* Step Content */}
        <Card className="mb-8">
          <CardContent className="p-6">
            {currentStep === 1 && <FolderSelection files={files} setFiles={setFiles} />}
            {currentStep === 2 && (
              <FileProcessing
                files={files}
                setFiles={setFiles}
                progress={processingProgress}
                setProgress={setProcessingProgress}
              />
            )}
            {currentStep === 3 && <FileSelectionForIngestion files={files} setFiles={setFiles} />}
            {currentStep === 4 && <DatabaseConfiguration config={databaseConfig} setConfig={setDatabaseConfig} />}
            {currentStep === 5 && (
              <IngestionProcess
                files={files}
                setFiles={setFiles}
                databaseConfig={databaseConfig}
                progress={ingestionProgress}
                setProgress={setIngestionProgress}
              />
            )}
            {currentStep === 6 && <SummaryView files={files} databaseConfig={databaseConfig} />}
          </CardContent>
        </Card>

        {/* Navigation */}
        <div className="flex justify-between">
          <Button
            variant="outline"
            onClick={prevStep}
            disabled={currentStep === 1}
            className="flex items-center gap-2 bg-transparent"
          >
            <ArrowLeft className="w-4 h-4" />
            Previous
          </Button>

          {currentStep < 6 ? (
            <Button onClick={nextStep} disabled={!canProceed()} className="flex items-center gap-2">
              Next
              <ArrowRight className="w-4 h-4" />
            </Button>
          ) : (
            <Button variant="default" className="flex items-center gap-2">
              <Download className="w-4 h-4" />
              Export Report
            </Button>
          )}
        </div>
      </div>
    </div>
  )
}
