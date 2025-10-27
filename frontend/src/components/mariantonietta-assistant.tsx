"use client"

import type React from "react"
import { useState, useRef } from "react"
import { Button } from "./ui/button"
import { Input } from "./ui/input"
import { Card } from "./ui/card"
import { Mic, ScanFace, Send, MicOff } from "lucide-react"
import AssistantCharacter from "./assistant-character"
import MessageList from "./message-list"

type Message = {
  id: string
  role: "user" | "assistant"
  content: string
  timestamp: Date
}

type AssistantState = "idle" | "listening" | "thinking" | "speaking"

export default function MariantoniettaAssistant() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [assistantState, setAssistantState] = useState<AssistantState>("idle")
  const [isRecording, setIsRecording] = useState(false)
  const [currentResponse, setCurrentResponse] = useState("")
  const inputRef = useRef<HTMLInputElement>(null)

  // Refs para poder detener/limpiar grabación
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const mediaStreamRef = useRef<MediaStream | null>(null)
  const stopTimerRef = useRef<number | null>(null)

  // ---- Grabación de voz y envío a /stt ----
  const handleVoiceRecording = async () => {
    if (!isRecording) {
      try {
        setIsRecording(true)
        setAssistantState("listening")

        const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
        mediaStreamRef.current = stream

        const mediaRecorder = new MediaRecorder(stream)
        mediaRecorderRef.current = mediaRecorder

        const audioChunks: Blob[] = []

        mediaRecorder.ondataavailable = (event) => {
          if (event.data && event.data.size > 0) audioChunks.push(event.data)
        }

        mediaRecorder.onerror = (e) => {
          console.error("MediaRecorder error:", e)
        }

        mediaRecorder.onstop = async () => {
          try {
            // Blob de webm/opus
            const audioBlob = new Blob(audioChunks, { type: "audio/webm" })

            // Convertir a WAV mono PCM16 a 16 kHz
            const wavBlob = await convertToWav(audioBlob)

            // Enviar al backend
            const formData = new FormData()
            formData.append("file", wavBlob, "audio.wav")

            const resp = await fetch("http://localhost:8000/stt", {
              method: "POST",
              body: formData,
            })

            if (!resp.ok) {
              const ct = resp.headers.get("content-type") || ""
              const errText = ct.includes("application/json")
                ? JSON.stringify(await resp.json())
                : await resp.text()
              console.error("STT error:", errText)
              throw new Error(errText || "Error al transcribir el audio")
            }

            const data = await resp.json()
            setInput(data.transcript || "")
          } catch (e) {
            console.error("Grabación/STT falló:", e)
          } finally {
            // Liberar el micro
            if (mediaStreamRef.current) {
              mediaStreamRef.current.getTracks().forEach((t) => t.stop())
              mediaStreamRef.current = null
            }
            mediaRecorderRef.current = null
            if (stopTimerRef.current) {
              window.clearTimeout(stopTimerRef.current)
              stopTimerRef.current = null
            }
            setIsRecording(false)
            setAssistantState("idle")
          }
        }

        mediaRecorder.start()

        // Detener automáticamente a los 5s (puedes ajustar)
        stopTimerRef.current = window.setTimeout(() => {
          if (mediaRecorder.state === "recording") mediaRecorder.stop()
        }, 5000)
      } catch (err) {
        console.error("No se pudo iniciar la grabación:", err)
        setIsRecording(false)
        setAssistantState("idle")
        // Limpiar por si quedó algo abierto
        if (mediaStreamRef.current) {
          mediaStreamRef.current.getTracks().forEach((t) => t.stop())
          mediaStreamRef.current = null
        }
        mediaRecorderRef.current = null
        if (stopTimerRef.current) {
          window.clearTimeout(stopTimerRef.current)
          stopTimerRef.current = null
        }
      }
    } else {
      // Si ya está grabando y el usuario vuelve a presionar, detenemos manualmente
      try {
        if (mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") {
          mediaRecorderRef.current.stop()
        }
      } finally {
        if (stopTimerRef.current) {
          window.clearTimeout(stopTimerRef.current)
          stopTimerRef.current = null
        }
        setIsRecording(false)
        setAssistantState("idle")
      }
    }
  }

  // ---- Conversión: webm/opus -> WAV mono PCM16 @ 16kHz ----
  const convertToWav = async (audioBlob: Blob): Promise<Blob> => {
    const arrayBuffer = await audioBlob.arrayBuffer()

    // 1) Decodificar a AudioBuffer (float32)
    const ctx = new (window.AudioContext || (window as any).webkitAudioContext)()
    const decoded: AudioBuffer = await ctx.decodeAudioData(arrayBuffer)

    // 2) Resample + downmix a mono y 16 kHz con OfflineAudioContext
    const targetSampleRate = 16000
    const lengthInSamples = Math.ceil(decoded.duration * targetSampleRate)
    const offline = new OfflineAudioContext(1, lengthInSamples, targetSampleRate)
    const src = offline.createBufferSource()
    src.buffer = decoded // el destino es mono; el motor hace downmix
    src.connect(offline.destination)
    src.start(0)
    const rendered: AudioBuffer = await offline.startRendering() // mono @ 16kHz

    // 3) Empaquetar a WAV PCM16 little-endian
    const channelData = rendered.getChannelData(0)
    const wavBuffer = floatTo16BitWav(channelData, targetSampleRate)
    return new Blob([wavBuffer], { type: "audio/wav" })
  }

  // Empaqueta Float32 mono a WAV PCM16 LE
  function floatTo16BitWav(float32: Float32Array, sampleRate: number): ArrayBuffer {
    const numChannels = 1
    const bytesPerSample = 2
    const blockAlign = numChannels * bytesPerSample
    const byteRate = sampleRate * blockAlign
    const dataSize = float32.length * bytesPerSample

    const buffer = new ArrayBuffer(44 + dataSize)
    const view = new DataView(buffer)

    // RIFF/WAVE header
    writeString(view, 0, "RIFF")
    view.setUint32(4, 36 + dataSize, true)
    writeString(view, 8, "WAVE")

    // fmt  subchunk
    writeString(view, 12, "fmt ")
    view.setUint32(16, 16, true) // Subchunk1Size
    view.setUint16(20, 1, true) // AudioFormat = PCM
    view.setUint16(22, numChannels, true) // NumChannels
    view.setUint32(24, sampleRate, true) // SampleRate
    view.setUint32(28, byteRate, true) // ByteRate
    view.setUint16(32, blockAlign, true) // BlockAlign
    view.setUint16(34, 16, true) // BitsPerSample

    // data subchunk
    writeString(view, 36, "data")
    view.setUint32(40, dataSize, true)

    // PCM 16-bit little endian
    let offset = 44
    for (let i = 0; i < float32.length; i++, offset += 2) {
      let s = Math.max(-1, Math.min(1, float32[i]))
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true)
    }

    return buffer
  }

  function writeString(view: DataView, offset: number, str: string) {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i))
    }
  }

  // ---- Cámara / reconocimiento facial (placeholder) ----
  const handleCamera = () => {
    setAssistantState("thinking")
    setTimeout(() => {
      const userMessage: Message = {
        id: Date.now().toString(),
        role: "user",
        content: "[Facial Recognition Activated]",
        timestamp: new Date(),
      }
      setMessages((prev) => [...prev, userMessage])
      simulateAssistantResponse("Hello! I recognize you. How can I help you today?")
    }, 1500)
  }

  // ---- Simulación de escritura de la respuesta ----
  const simulateAssistantResponse = (response: string) => {
    setAssistantState("thinking")
    setCurrentResponse("")

    setTimeout(() => {
      setAssistantState("speaking")
      let index = 0
      const typingInterval = setInterval(() => {
        if (index < response.length) {
          setCurrentResponse((prev) => prev + response[index])
          index++
        } else {
          clearInterval(typingInterval)
          const assistantMessage: Message = {
            id: Date.now().toString(),
            role: "assistant",
            content: response,
            timestamp: new Date(),
          }
          setMessages((prev) => [...prev, assistantMessage])
          setCurrentResponse("")
          setAssistantState("idle")
        }
      }, 30)
    }, 1000)
  }

  // ---- Envío de texto a /ask ----
  const handleSend = async () => {
    if (!input.trim()) return

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input,
      timestamp: new Date(),
    }

    setMessages((prev) => [...prev, userMessage])
    setInput("")
    setAssistantState("thinking")
    setCurrentResponse("")

    try {
      const response = await fetch("http://localhost:8000/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query: input }),
      })

      if (!response.ok) {
        const errText = await response.text()
        throw new Error(errText || "Error en la comunicación con el backend")
      }

      const data = await response.json()
      setAssistantState("speaking")
      setCurrentResponse(data.respuesta)

      const assistantMessage: Message = {
        id: Date.now().toString(),
        role: "assistant",
        content: data.respuesta,
        timestamp: new Date(),
      }

      setMessages((prev) => [...prev, assistantMessage])
      setCurrentResponse("")
      setAssistantState("idle")
    } catch (error) {
      console.error("Error:", error)
      setAssistantState("idle")
    }
  }

  // Enter para enviar
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="flex flex-col h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card/50 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-4">
          <h1 className="text-2xl font-bold text-foreground">Mariantonieta</h1>
          <p className="text-sm text-muted-foreground">Your AI Voice Assistant</p>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-1 container mx-auto px-4 py-6 flex flex-col lg:flex-row gap-6 overflow-hidden">
        {/* Character Section */}
        <div className="lg:w-1/3 flex items-center justify-center">
          <AssistantCharacter state={assistantState} />
        </div>

        {/* Chat Section */}
        <div className="lg:w-2/3 flex flex-col gap-4 min-h-0">
          {/* Messages */}
          <Card className="flex-1 p-4 overflow-hidden flex flex-col bg-card/50 backdrop-blur-sm">
            <MessageList messages={messages} currentResponse={currentResponse} assistantState={assistantState} />
          </Card>

          {/* Input Area */}
          <Card className="p-4 bg-card/50 backdrop-blur-sm">
            <div className="flex gap-2 items-end">
              {/* Camera Button */}
              <Button
                size="icon"
                variant="outline"
                onClick={handleCamera}
                disabled={assistantState !== "idle"}
                className="shrink-0 h-12 w-12 rounded-xl border-primary/20 hover:border-primary hover:bg-primary/10 bg-transparent"
                title="Facial Recognition"
              >
                <ScanFace className="h-5 w-5" />
              </Button>

              {/* Voice Button */}
              <Button
                size="icon"
                variant={isRecording ? "default" : "outline"}
                onClick={handleVoiceRecording}
                disabled={assistantState === "thinking" || assistantState === "speaking"}
                className={`shrink-0 h-12 w-12 rounded-xl ${
                  isRecording
                    ? "bg-accent hover:bg-accent/90 animate-pulse-glow"
                    : "border-primary/20 hover:border-primary hover:bg-primary/10"
                }`}
                title={isRecording ? "Stop recording" : "Start recording"}
              >
                {isRecording ? <MicOff className="h-5 w-5" /> : <Mic className="h-5 w-5" />}
              </Button>

              {/* Text Input */}
              <div className="flex-1 flex gap-2">
                <Input
                  ref={inputRef}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Type your message or use voice..."
                  disabled={assistantState === "thinking" || assistantState === "speaking"}
                  className="h-12 bg-secondary/50 border-primary/20 focus:border-primary rounded-xl"
                />
                <Button
                  size="icon"
                  onClick={handleSend}
                  disabled={!input.trim() || assistantState === "thinking" || assistantState === "speaking"}
                  className="shrink-0 h-12 w-12 rounded-xl bg-primary hover:bg-primary/90"
                >
                  <Send className="h-5 w-5" />
                </Button>
              </div>
            </div>

            {/* Status Indicator */}
            {assistantState !== "idle" && (
              <div className="mt-3 text-sm text-muted-foreground flex items-center gap-2">
                <div className="h-2 w-2 rounded-full bg-primary animate-pulse" />
                {assistantState === "listening" && "Listening..."}
                {assistantState === "thinking" && "Thinking..."}
                {assistantState === "speaking" && "Speaking..."}
              </div>
            )}
          </Card>
        </div>
      </div>
    </div>
  )
}