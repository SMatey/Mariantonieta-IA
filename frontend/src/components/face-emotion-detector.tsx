"use client"

import { useState, useRef, useEffect } from "react"
import { Button } from "./ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card"
import { Camera, X, Loader2 } from "lucide-react"

type EmotionResult = {
  main_emotion: string
  score: number
  position: {
    left: number
    top: number
    width: number
    height: number
  }
}

type EmotionAnalysis = {
  success: boolean
  faces_detected: number
  results: EmotionResult[]
  message?: string
}

interface FaceEmotionDetectorProps {
  onClose?: () => void
  onEmotionDetected?: (emotion: string) => void
}

export default function FaceEmotionDetector({ onClose, onEmotionDetected }: FaceEmotionDetectorProps) {
  const [isStreaming, setIsStreaming] = useState(false)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [emotionData, setEmotionData] = useState<EmotionAnalysis | null>(null)
  const [error, setError] = useState<string | null>(null)

  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const streamRef = useRef<MediaStream | null>(null)

  // Iniciar la cámara
  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
      })

      if (videoRef.current) {
        videoRef.current.srcObject = stream
        streamRef.current = stream
        setIsStreaming(true)
        setError(null)
      }
    } catch (err) {
      setError("No se pudo acceder a la cámara. Por favor, verifica los permisos.")
      console.error("Error al acceder a la cámara:", err)
    }
  }

  // Detener la cámara
  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop())
      streamRef.current = null
    }
    setIsStreaming(false)
  }

  // Capturar foto y analizar emoción
  const captureAndAnalyze = async () => {
    if (!videoRef.current || !canvasRef.current) return

    setIsAnalyzing(true)
    setError(null)

    try {
      const video = videoRef.current
      const canvas = canvasRef.current
      const context = canvas.getContext("2d")

      if (!context) return

      // Configurar el canvas con las dimensiones del video
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight

      // Dibujar el frame actual del video en el canvas
      context.drawImage(video, 0, 0, canvas.width, canvas.height)

      // Convertir el canvas a blob
      const blob = await new Promise<Blob>((resolve) => {
        canvas.toBlob(
          (blob) => {
            if (blob) resolve(blob)
          },
          "image/jpeg",
          0.95,
        )
      })

      // Enviar la imagen al backend
      const formData = new FormData()
      formData.append("file", blob, "capture.jpg")

      const response = await fetch("http://localhost:8000/face/analyze-emotion", {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        throw new Error("Error al analizar la imagen")
      }

      const data: EmotionAnalysis = await response.json()
      setEmotionData(data)

      // Notificar la emoción detectada
      if (data.success && data.results.length > 0 && onEmotionDetected) {
        onEmotionDetected(data.results[0].main_emotion)
      }
    } catch (err) {
      setError("Error al analizar la emoción. Por favor, intenta de nuevo.")
      console.error("Error:", err)
    } finally {
      setIsAnalyzing(false)
    }
  }

  // Limpiar al desmontar
  useEffect(() => {
    return () => {
      stopCamera()
    }
  }, [])

  // Mapeo de emociones a emojis
  const getEmotionEmoji = (emotion: string) => {
    const emojiMap: Record<string, string> = {
      happiness: "😊",
      sadness: "😢",
      anger: "😠",
      surprise: "😲",
      fear: "😨",
      disgust: "🤢",
      contempt: "😒",
      neutral: "😐",
    }
    return emojiMap[emotion.toLowerCase()] || "😐"
  }

  return (
    <Card className="w-full max-w-2xl mx-auto">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>Detección de Emociones</CardTitle>
            <CardDescription>Usa tu cámara para detectar emociones faciales en tiempo real</CardDescription>
          </div>
          {onClose && (
            <Button variant="ghost" size="icon" onClick={onClose}>
              <X className="h-5 w-5" />
            </Button>
          )}
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Video Preview */}
        <div className="relative bg-secondary rounded-lg overflow-hidden aspect-video">
          <video ref={videoRef} autoPlay playsInline muted className="w-full h-full object-cover" />
          <canvas ref={canvasRef} className="hidden" />

          {!isStreaming && (
            <div className="absolute inset-0 flex items-center justify-center bg-secondary/80">
              <div className="text-center space-y-4">
                <Camera className="h-16 w-16 mx-auto text-muted-foreground" />
                <p className="text-muted-foreground">Cámara desactivada</p>
              </div>
            </div>
          )}
        </div>

        {/* Error Message */}
        {error && (
          <div className="p-4 bg-destructive/10 border border-destructive rounded-lg">
            <p className="text-sm text-destructive">{error}</p>
          </div>
        )}

        {/* Emotion Results */}
        {emotionData && emotionData.success && (
          <div className="p-4 bg-primary/10 border border-primary rounded-lg space-y-2">
            <h3 className="font-semibold text-foreground">
              {emotionData.faces_detected}{" "}
              {emotionData.faces_detected === 1 ? "rostro detectado" : "rostros detectados"}
            </h3>
            {emotionData.results.map((result, index) => (
              <div key={index} className="flex items-center gap-3 p-3 bg-card rounded-md">
                <span className="text-4xl">{getEmotionEmoji(result.main_emotion)}</span>
                <div className="flex-1">
                  <p className="font-medium capitalize">{result.main_emotion}</p>
                  <p className="text-sm text-muted-foreground">Confianza: {(result.score * 100).toFixed(1)}%</p>
                </div>
              </div>
            ))}
          </div>
        )}

        {emotionData && !emotionData.success && (
          <div className="p-4 bg-muted rounded-lg">
            <p className="text-sm text-muted-foreground">{emotionData.message}</p>
          </div>
        )}

        {/* Controls */}
        <div className="flex gap-2">
          {!isStreaming ? (
            <Button onClick={startCamera} className="flex-1">
              <Camera className="mr-2 h-4 w-4" />
              Activar Cámara
            </Button>
          ) : (
            <>
              <Button onClick={captureAndAnalyze} disabled={isAnalyzing} className="flex-1">
                {isAnalyzing ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Analizando...
                  </>
                ) : (
                  <>
                    <Camera className="mr-2 h-4 w-4" />
                    Capturar y Analizar
                  </>
                )}
              </Button>
              <Button onClick={stopCamera} variant="outline">
                Detener
              </Button>
            </>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
