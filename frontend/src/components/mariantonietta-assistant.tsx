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

  // Grabar audio y convertirlo a WAV usando Web Audio API
  const handleVoiceRecording = async () => {
    if (!isRecording) {
      setIsRecording(true)
      setAssistantState("listening")

      // Graba audio
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mediaRecorder = new MediaRecorder(stream)

      let audioChunks: Blob[] = []

      mediaRecorder.ondataavailable = (event) => {
        audioChunks.push(event.data)
      }

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunks, { type: "audio/webm" })
        const audioBuffer = await audioBlob.arrayBuffer()

        // Convertimos a WAV
        const wavBlob = await convertToWav(audioBuffer)

        // Enviamos el archivo WAV al backend
        const formData = new FormData()
        formData.append("file", wavBlob, "audio.wav")

        const response = await fetch("http://localhost:8000/stt", {
          method: "POST",
          body: formData,
        })

        if (response.ok) {
          const data = await response.json()
          // Actualizamos el input con la transcripción
          setInput(data.transcript)
        } else {
          console.error("Error al transcribir el audio.")
        }

        // Resetear el estado después de la grabación
        setIsRecording(false)
        setAssistantState("idle")
      }

      mediaRecorder.start()

      // Detenemos la grabación después de 5 segundos (puedes cambiar esto)
      setTimeout(() => {
        mediaRecorder.stop()
      }, 5000)
    } else {
      setIsRecording(false)
      setAssistantState("idle")
    }
  }

  // Convertir el buffer de audio a WAV usando la Web Audio API
  const convertToWav = async (audioBuffer: ArrayBuffer) => {
    const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)()
    const buffer = await audioContext.decodeAudioData(audioBuffer)

    const wavData = encodeWAV(buffer)

    // Crear un Blob con los datos WAV y devolverlo
    const wavBlob = new Blob([wavData], { type: "audio/wav" })
    return wavBlob
  }

  // Función que convierte el buffer de audio a formato WAV
  const encodeWAV = (buffer: AudioBuffer) => {
    const numChannels = buffer.numberOfChannels
    const sampleRate = buffer.sampleRate
    const samples = buffer.length

    const bufferArray = new ArrayBuffer(44 + samples * 2 * numChannels) // 44 bytes para el encabezado WAV
    const view = new DataView(bufferArray)

    // Escribir encabezado WAV
    writeString(view, 0, "RIFF")
    view.setUint32(4, 36 + samples * 2 * numChannels, true)
    writeString(view, 8, "WAVE")
    writeString(view, 12, "fmt ")
    view.setUint32(16, 16, true) // Subchunk1Size
    view.setUint16(20, 1, true) // AudioFormat (PCM)
    view.setUint16(22, numChannels, true)
    view.setUint32(24, sampleRate, true)
    view.setUint32(28, sampleRate * numChannels * 2, true) // ByteRate
    view.setUint16(32, numChannels * 2, true) // BlockAlign
    view.setUint16(34, 16, true) // BitsPerSample
    writeString(view, 36, "data")
    view.setUint32(40, samples * 2 * numChannels, true) // Subchunk2Size

    // Escribir datos de audio
    let offset = 44
    for (let channel = 0; channel < numChannels; channel++) {
      const channelData = buffer.getChannelData(channel)
      for (let i = 0; i < samples; i++) {
        const sample = Math.max(-1, Math.min(1, channelData[i])) // Normalizar entre -1 y 1
        view.setInt16(offset, sample * 0x7fff, true) // 16 bits PCM
        offset += 2
      }
    }

    return new Uint8Array(bufferArray)
  }

  // Función auxiliar para escribir cadenas en el ArrayBuffer
  const writeString = (view: DataView, offset: number, str: string) => {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i))
    }
  }

  // Handle camera/facial recognition
  const handleCamera = () => {
    setAssistantState("thinking")

    // Simulate facial recognition processing
    setTimeout(() => {
      const userMessage: Message = {
        id: Date.now().toString(),
        role: "user",
        content: "[Facial Recognition Activated]",
        timestamp: new Date(),
      }
      setMessages((prev) => [...prev, userMessage])

      // Simulate assistant response
      simulateAssistantResponse("Hello! I recognize you. How can I help you today?")
    }, 1500)
  }

  // Simulate typing animation for assistant response
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

          // Add complete message to history
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

  // Handle sending message
  const handleSend = () => {
    if (!input.trim()) return

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input,
      timestamp: new Date(),
    }

    setMessages((prev) => [...prev, userMessage])
    setInput("")

    // Simulate assistant response
    const responses = [
      "I'm here to help! What would you like to know?",
      "That's an interesting question. Let me think about that...",
      "I understand. Here's what I can tell you about that.",
      "Great question! I'd be happy to assist you with that.",
      "I'm processing your request. One moment please.",
    ]
    const randomResponse = responses[Math.floor(Math.random() * responses.length)]
    simulateAssistantResponse(randomResponse)
  }

  // Handle Enter key
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
