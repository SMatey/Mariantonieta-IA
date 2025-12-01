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
  // ¡NUEVOS CAMPOS PARA LAS PROBABILIDADES DETALLADAS!
  detailed_probabilities?: {
    angry: number
    disgust: number
    fear: number
    happy: number
    neutral: number
    sad: number
    surprise: number
  }
  raw_prediction?: string
  model_info?: {
    architecture: string
    confidence: number
    total_classes: number
  }
}

type GoogleVisionResponse = {
  faces: FaceResult[]
  meta: {
    source: string
    notes: string
    saved_image?: string
    timestamp?: string
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

  // Componente para mostrar las probabilidades detalladas
  const DetailedProbabilities = ({ probabilities }: { probabilities: FaceResult['detailed_probabilities'] }) => {
    if (!probabilities) return null;

    // Ordenar emociones por probabilidad (mayor a menor)
    const sortedEmotions = Object.entries(probabilities)
      .sort(([, a], [, b]) => b - a);

    const getEmotionLabel = (emotion: string) => {
      const labels: Record<string, string> = {
        angry: "Enojo",
        disgust: "Asco", 
        fear: "Miedo",
        happy: "Felicidad",
        neutral: "Neutral",
        sad: "Tristeza",
        surprise: "Sorpresa"
      }
      return labels[emotion] || emotion;
    };

    const getEmotionColor = (emotion: string) => {
      const colors: Record<string, string> = {
        angry: "bg-red-500",
        disgust: "bg-green-500",
        fear: "bg-purple-500", 
        happy: "bg-yellow-500",
        neutral: "bg-gray-500",
        sad: "bg-blue-500",
        surprise: "bg-orange-500"
      }
      return colors[emotion] || "bg-gray-500";
    };

    return (
      <div className="mt-4 p-4 bg-gray-50 rounded-lg border">
        <h4 className="font-semibold text-gray-800 mb-3 text-center">📊 Probabilidades por clase:</h4>
        <div className="space-y-3">
          {sortedEmotions.map(([emotion, probability]) => (
            <div key={emotion} className="flex items-center justify-between">
              <span className="text-sm font-medium text-gray-700 capitalize w-20">
                {getEmotionLabel(emotion)}:
              </span>
              <div className="flex-1 mx-3 bg-gray-200 rounded-full h-3 overflow-hidden">
                <div 
                  className={`h-3 rounded-full transition-all duration-500 ${getEmotionColor(emotion)}`}
                  style={{ width: `${probability * 100}%` }}
                ></div>
              </div>
              <span className="text-sm text-gray-800 font-mono min-w-[65px] text-right">
                {(probability).toFixed(4)}
              </span>
            </div>
          ))}
        </div>
        <div className="mt-3 text-xs text-gray-500 text-center">
          ↑ Ordenado por probabilidad (mayor a menor)
        </div>
      </div>
    );
  }

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

      // Mostrar notificación si la imagen se guardó
      if (data.meta?.saved_image) {
        console.log("📸 Imagen guardada en:", data.meta.saved_image)
      }

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
      // Mapeos para las 7 emociones del modelo FER2013
      angry: "😠",
      happy: "😊",
      sad: "😢"
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
      // Traducciones para las 7 emociones FER2013
      angry: "Enojo",
      disgust: "Asco",
      fear: "Miedo", 
      happy: "Felicidad",
      sad: "Tristeza"
    }
    return translations[emotion.toLowerCase()] || emotion
  }

  return (
    <Card className="w-full max-w-2xl mx-auto max-h-[90vh] flex flex-col">
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

      <CardContent className="space-y-4 max-h-[70vh] overflow-y-auto">
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
          <div className="p-4 bg-primary/10 border border-primary rounded-lg space-y-2 max-h-96 overflow-y-auto">
            <h3 className="font-semibold text-foreground">
              {emotionData.faces.length} {emotionData.faces.length === 1 ? "rostro detectado" : "rostros detectados"}
            </h3>
            {emotionData.meta?.saved_image && (
              <div className="text-xs text-muted-foreground bg-green-50 p-2 rounded border border-green-200">
                📸 Imagen guardada: {emotionData.meta.saved_image.split('/').pop()}
              </div>
            )}
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
                
                {/* Mostrar probabilidades detalladas si están disponibles */}
                <DetailedProbabilities probabilities={face.detailed_probabilities} />
                
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
