"use client"

import { useEffect, useRef } from "react"
import { Card } from "./ui/card"

type Message = {
  id: string
  role: "user" | "assistant"
  content: string
  timestamp: Date
}

type AssistantState = "idle" | "listening" | "thinking" | "speaking"

interface MessageListProps {
  messages: Message[]
  currentResponse: string
  assistantState: AssistantState
}

export default function MessageList({ messages, currentResponse, assistantState }: MessageListProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages, currentResponse])

  return (
    <div className="flex-1 overflow-y-auto space-y-4 pr-2">
      {messages.length === 0 && !currentResponse && (
        <div className="h-full flex items-center justify-center">
          <div className="text-center space-y-2">
            <h3 className="text-xl font-semibold text-foreground">Welcome!</h3>
            <p className="text-muted-foreground max-w-md">
              I'm Mariantonieta, your AI assistant. You can type a message, use voice commands, or activate facial
              recognition to get started.
            </p>
          </div>
        </div>
      )}

      {messages.map((message) => (
        <div key={message.id} className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}>
          <Card
            className={`max-w-[80%] p-4 ${
              message.role === "user" ? "bg-primary text-primary-foreground" : "bg-secondary text-secondary-foreground"
            }`}
          >
            <p className="text-sm leading-relaxed">{message.content}</p>
            <p className="text-xs opacity-70 mt-2">
              {message.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
            </p>
          </Card>
        </div>
      ))}

      {/* Current Response (Typing Animation) */}
      {currentResponse && (
        <div className="flex justify-start">
          <Card className="max-w-[80%] p-4 bg-secondary text-secondary-foreground">
            <p className="text-sm leading-relaxed">
              {currentResponse}
              <span className="inline-block w-1 h-4 bg-foreground ml-1 animate-pulse" />
            </p>
          </Card>
        </div>
      )}

      {/* Thinking Indicator */}
      {assistantState === "thinking" && !currentResponse && (
        <div className="flex justify-start">
          <Card className="p-4 bg-secondary text-secondary-foreground">
            <div className="flex gap-2 items-center">
              <div className="flex gap-1">
                {[0, 1, 2].map((i) => (
                  <div
                    key={i}
                    className="w-2 h-2 rounded-full bg-foreground/50 animate-bounce"
                    style={{ animationDelay: `${i * 0.15}s` }}
                  />
                ))}
              </div>
              <span className="text-sm text-muted-foreground">Thinking...</span>
            </div>
          </Card>
        </div>
      )}

      <div ref={messagesEndRef} />
    </div>
  )
}