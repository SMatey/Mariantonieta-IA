"use client"

import { useState, useRef, type ChangeEvent } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card"
import { Button } from "./ui/button"
import { Upload, X, Loader2, Car } from "lucide-react"

interface BoundingBox {
  x1: number
  y1: number
  x2: number
  y2: number
}

interface VehicleDetection {
  class_name: string
  confidence: number
  bounding_box: BoundingBox
}

interface ImageAnalysisResult {
  detections: VehicleDetection[]
  total_vehicles: number
  vehicle_counts: Record<string, number>
  processing_time_ms: number
  image_size: { width: number; height: number }
  saved_image?: string
}

interface VideoAnalysisResult {
  total_frames: number
  processed_frames: number
  average_vehicles_per_frame: number
  total_unique_detections: number
  vehicle_counts: Record<string, number>
  processing_time_ms: number
}

interface VehicleDetectionProps {
  onClose?: () => void
}

export default function VehicleDetection({ onClose }: VehicleDetectionProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [fileType, setFileType] = useState<"image" | "video" | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string>("")
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [imageResult, setImageResult] = useState<ImageAnalysisResult | null>(null)
  const [videoResult, setVideoResult] = useState<VideoAnalysisResult | null>(null)
  const [error, setError] = useState<string>("")
  const [maxFrames, setMaxFrames] = useState(30)

  const fileInputRef = useRef<HTMLInputElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  const handleFileSelect = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    const isImage = file.type.startsWith("image/")
    const isVideo = file.type.startsWith("video/")

    if (!isImage && !isVideo) {
      setError("Por favor selecciona una imagen o video válido")
      return
    }

    setSelectedFile(file)
    setFileType(isImage ? "image" : "video")
    setError("")
    setImageResult(null)
    setVideoResult(null)

    const url = URL.createObjectURL(file)
    setPreviewUrl(url)
  }

  const analyzeImage = async () => {
    if (!selectedFile) return

    setIsAnalyzing(true)
    setError("")

    try {
      const formData = new FormData()
      formData.append("file", selectedFile)

      const response = await fetch("http://localhost:8000/vehicles/analyze-image", {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`Error ${response.status}: ${await response.text()}`)
      }

      const result: ImageAnalysisResult = await response.json()
      setImageResult(result)
      drawDetections(result)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Error analizando imagen")
    } finally {
      setIsAnalyzing(false)
    }
  }

  const analyzeVideo = async () => {
    if (!selectedFile) return

    setIsAnalyzing(true)
    setError("")

    try {
      const formData = new FormData()
      formData.append("file", selectedFile)

      const response = await fetch(
        `http://localhost:8000/vehicles/analyze-video?max_frames=${maxFrames}`,
        {
          method: "POST",
          body: formData,
        },
      )

      if (!response.ok) {
        throw new Error(`Error ${response.status}: ${await response.text()}`)
      }

      const result: VideoAnalysisResult = await response.json()
      setVideoResult(result)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Error analizando video")
    } finally {
      setIsAnalyzing(false)
    }
  }

  const drawDetections = (result: ImageAnalysisResult) => {
    const canvas = canvasRef.current
    if (!canvas) return

    const img = new Image()
    img.onload = () => {
      canvas.width = img.width
      canvas.height = img.height

      const ctx = canvas.getContext("2d")
      if (!ctx) return

      ctx.drawImage(img, 0, 0)

      result.detections.forEach((detection) => {
        const { x1, y1, x2, y2 } = detection.bounding_box

        ctx.strokeStyle = "#00ff00"
        ctx.lineWidth = 3
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)

        const label = `${detection.class_name} ${(detection.confidence * 100).toFixed(0)}%`
        ctx.fillStyle = "#00ff00"
        ctx.font = "16px Arial"
        ctx.fillText(label, x1, y1 - 5)
      })
    }

    img.src = previewUrl
  }

  const handleAnalyze = () => {
    if (fileType === "image") {
      analyzeImage()
    } else if (fileType === "video") {
      analyzeVideo()
    }
  }

  const resetAnalysis = () => {
    setSelectedFile(null)
    setFileType(null)
    setPreviewUrl("")
    setImageResult(null)
    setVideoResult(null)
    setError("")
    if (fileInputRef.current) {
      fileInputRef.current.value = ""
    }
  }

  const getVehicleEmoji = (type: string) => {
    const emojiMap: Record<string, string> = {
      car: "🚗",
      truck: "🚚",
      bus: "🚌",
      motorcycle: "🏍️",
    }
    return emojiMap[type.toLowerCase()] || "🚗"
  }

  return (
    <Card className="w-full max-w-6xl mx-auto max-h-[90vh] flex flex-col">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>Detección de Vehículos</CardTitle>
            <CardDescription>Analiza imágenes y videos para detectar vehículos automáticamente</CardDescription>
          </div>
          {onClose && (
            <Button variant="ghost" size="icon" onClick={onClose}>
              <X className="h-5 w-5" />
            </Button>
          )}
        </div>
      </CardHeader>

      <CardContent className="space-y-4 max-h-[75vh] overflow-y-auto">
        <div className="flex gap-2">
          <Button onClick={() => fileInputRef.current?.click()} className="flex-1" variant="outline">
            <Upload className="mr-2 h-4 w-4" />
            Seleccionar Archivo
          </Button>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*,video/*"
            onChange={handleFileSelect}
            className="hidden"
          />
        </div>

        {selectedFile && (
          <div className="text-sm text-muted-foreground bg-muted p-2 rounded">
            📎 {selectedFile.name} ({fileType === "image" ? "Imagen" : "Video"})
          </div>
        )}

        {fileType === "video" && (
          <div className="space-y-2">
            <label className="block text-sm font-medium">Frames a procesar: {maxFrames}</label>
            <input
              type="range"
              min="1"
              max="100"
              value={maxFrames}
              onChange={(e) => setMaxFrames(parseInt(e.target.value))}
              className="w-full"
            />
            <p className="text-xs text-muted-foreground">
              Más frames = más precisión pero más tiempo de procesamiento
            </p>
          </div>
        )}

        <div className="flex gap-2">
          <Button onClick={handleAnalyze} disabled={!selectedFile || isAnalyzing} className="flex-1">
            {isAnalyzing ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Analizando...
              </>
            ) : (
              <>
                <Car className="mr-2 h-4 w-4" />
                Analizar
              </>
            )}
          </Button>
          <Button onClick={resetAnalysis} variant="outline" disabled={isAnalyzing}>
            Nuevo Análisis
          </Button>
        </div>

        {error && (
          <div className="p-4 bg-destructive/10 border border-destructive rounded-lg">
            <p className="text-sm text-destructive">{error}</p>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div>
            <h3 className="text-lg font-semibold mb-3">Vista Previa</h3>
            {fileType === "image" && previewUrl && (
              <div className="relative bg-secondary rounded-lg overflow-hidden">
                <img
                  src={previewUrl}
                  alt="Preview"
                  className="w-full rounded"
                  style={{ display: imageResult ? "none" : "block" }}
                />
                <canvas
                  ref={canvasRef}
                  className="w-full rounded"
                  style={{ display: imageResult ? "block" : "none" }}
                />
              </div>
            )}
            {fileType === "video" && previewUrl && (
              <video src={previewUrl} controls className="w-full rounded bg-secondary" />
            )}
            {!previewUrl && (
              <div className="bg-secondary rounded-lg h-64 flex items-center justify-center">
                <p className="text-muted-foreground">No hay archivo seleccionado</p>
              </div>
            )}
          </div>

          <div>
            <h3 className="text-lg font-semibold mb-3">Resultados</h3>

            {imageResult && (
              <div className="space-y-4">
                <div className="p-4 bg-primary/10 border border-primary rounded-lg">
                  <p className="text-2xl font-bold text-primary">{imageResult.total_vehicles} vehículos detectados</p>
                  <p className="text-sm text-muted-foreground">Tiempo: {imageResult.processing_time_ms.toFixed(0)}ms</p>
                </div>

                <div>
                  <h4 className="font-semibold mb-2">Por tipo:</h4>
                  <div className="space-y-2">
                    {Object.entries(imageResult.vehicle_counts).map(([type, count]) => (
                      <div key={type} className="flex items-center justify-between p-2 bg-card rounded">
                        <span className="flex items-center gap-2">
                          <span className="text-2xl">{getVehicleEmoji(type)}</span>
                          <span className="capitalize">{type}</span>
                        </span>
                        <span className="font-semibold">{count}</span>
                      </div>
                    ))}
                  </div>
                </div>

                <div>
                  <h4 className="font-semibold mb-2">Detecciones:</h4>
                  <div className="max-h-64 overflow-y-auto space-y-2">
                    {imageResult.detections.map((det, idx) => (
                      <div key={idx} className="p-2 bg-card rounded text-sm">
                        <span className="font-medium capitalize">{det.class_name}</span>
                        <span className="text-muted-foreground ml-2">{(det.confidence * 100).toFixed(1)}%</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {videoResult && (
              <div className="space-y-4">
                <div className="p-4 bg-primary/10 border border-primary rounded-lg">
                  <p className="text-2xl font-bold text-primary">{videoResult.total_unique_detections} detecciones</p>
                  <p className="text-sm text-muted-foreground">
                    {videoResult.average_vehicles_per_frame.toFixed(1)} vehículos/frame promedio
                  </p>
                  <p className="text-sm text-muted-foreground">
                    {videoResult.processed_frames}/{videoResult.total_frames} frames procesados
                  </p>
                  <p className="text-sm text-muted-foreground">
                    Tiempo: {(videoResult.processing_time_ms / 1000).toFixed(1)}s
                  </p>
                </div>

                <div>
                  <h4 className="font-semibold mb-2">Conteo por tipo:</h4>
                  <div className="space-y-2">
                    {Object.entries(videoResult.vehicle_counts).map(([type, count]) => (
                      <div key={type} className="flex items-center justify-between p-2 bg-card rounded">
                        <span className="flex items-center gap-2">
                          <span className="text-2xl">{getVehicleEmoji(type)}</span>
                          <span className="capitalize">{type}</span>
                        </span>
                        <span className="font-semibold">{count}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {!imageResult && !videoResult && !isAnalyzing && (
              <div className="bg-muted rounded-lg p-8 text-center">
                <p className="text-muted-foreground">Selecciona un archivo y haz clic en "Analizar"</p>
              </div>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
