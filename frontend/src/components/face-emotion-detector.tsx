"use client"

import { useState, useRef, useEffect } from "react"
import { Button } from "./ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card"
import { Camera, X, Loader2 } from "lucide-react"

type FaceResult = {
  detection_source: string
  position: {
    left: number
    top: number
    width: number
    height: number
  }
  likelihoods: {
    joy: string
    sorrow: string
    anger: string
    surprise: string
  }
  best_emotion: {
    label: string
    score: number
  }
}

type GoogleVisionResponse = {
  faces: FaceResult[]
  meta: {
    source: string
    notes: string
  }
}

interface FaceEmotionDetectorProps {
  onClose?: () => void
  onEmotionDetected?: (emotion: string) => void
}

export default function FaceEmotionDetector({ onClose, onEmotionDetected }: FaceEmotionDetectorProps) {
  const [isStreaming, setIsStreaming] = useState(false)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [emotionData, setEmotionData] = useState<GoogleVisionResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const streamRef = useRef<MediaStream | null>(null)

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

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop())
      streamRef.current = null
    }
    setIsStreaming(false)
  }

  const captureAndAnalyze = async () => {
    if (!videoRef.current || !canvasRef.current) return

    setIsAnalyzing(true)
    setError(null)

    try {
      const video = videoRef.current
      const canvas = canvasRef.current
      const context = canvas.getContext("2d")

      if (!context) return

      canvas.width = video.videoWidth
      canvas.height = video.videoHeight

      context.drawImage(video, 0, 0, canvas.width, canvas.height)

      const blob = await new Promise<Blob>((resolve) => {
        canvas.toBlob(
          (blob) => {
            if (blob) resolve(blob)
          },
          "image/jpeg",
          0.95,
        )
      })

      const formData = new FormData()
      formData.append("file", blob, "capture.jpg")

      const response = await fetch("http://localhost:8000/face/analyze", {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        throw new Error("Error al analizar la imagen")
      }

      const data: GoogleVisionResponse = await response.json()
      setEmotionData(data)

      if (data.faces.length > 0 && onEmotionDetected) {
        onEmotionDetected(data.faces[0].best_emotion.label)
      }
    } catch (err) {
      setError("Error al analizar la emoción. Por favor, intenta de nuevo.")
      console.error("Error:", err)
    } finally {
      setIsAnalyzing(false)
    }
  }

  useEffect(() => {
    return () => {
      stopCamera()
    }
  }, [])

  const getEmotionEmoji = (emotion: string) => {
    const emojiMap: Record<string, string> = {
      joy: "😊",
      sorrow: "😢",
      anger: "😠",
      surprise: "😲",
      neutral: "😐",
      happiness: "😊",
      sadness: "😢",
      fear: "😨",
      disgust: "🤢",
      contempt: "😒",
    }
    return emojiMap[emotion.toLowerCase()] || "😐"
  }

  const translateEmotion = (emotion: string) => {
    const translations: Record<string, string> = {
      joy: "Alegría",
      sorrow: "Tristeza",
      anger: "Enojo",
      surprise: "Sorpresa",
      neutral: "Neutral",
    }
    return translations[emotion.toLowerCase()] || emotion
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

        {error && (
          <div className="p-4 bg-destructive/10 border border-destructive rounded-lg">
            <p className="text-sm text-destructive">{error}</p>
          </div>
        )}

        {emotionData && emotionData.faces.length > 0 && (
          <div className="p-4 bg-primary/10 border border-primary rounded-lg space-y-2">
            <h3 className="font-semibold text-foreground">
              {emotionData.faces.length} {emotionData.faces.length === 1 ? "rostro detectado" : "rostros detectados"}
            </h3>
            {emotionData.faces.map((face, index) => (
              <div key={index} className="space-y-2">
                <div className="flex items-center gap-3 p-3 bg-card rounded-md">
                  <span className="text-4xl">{getEmotionEmoji(face.best_emotion.label)}</span>
                  <div className="flex-1">
                    <p className="font-medium">{translateEmotion(face.best_emotion.label)}</p>
                    <p className="text-sm text-muted-foreground">
                      Confianza: {(face.best_emotion.score * 100).toFixed(1)}%
                    </p>
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div className="p-2 bg-card rounded">
                    <span className="text-muted-foreground">Alegría:</span>{" "}
                    <span className="font-medium">{face.likelihoods.joy}</span>
                  </div>
                  <div className="p-2 bg-card rounded">
                    <span className="text-muted-foreground">Tristeza:</span>{" "}
                    <span className="font-medium">{face.likelihoods.sorrow}</span>
                  </div>
                  <div className="p-2 bg-card rounded">
                    <span className="text-muted-foreground">Enojo:</span>{" "}
                    <span className="font-medium">{face.likelihoods.anger}</span>
                  </div>
                  <div className="p-2 bg-card rounded">
                    <span className="text-muted-foreground">Sorpresa:</span>{" "}
                    <span className="font-medium">{face.likelihoods.surprise}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {emotionData && emotionData.faces.length === 0 && (
          <div className="p-4 bg-muted rounded-lg">
            <p className="text-sm text-muted-foreground">No se detectaron rostros en la imagen. Intenta de nuevo.</p>
          </div>
        )}

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
