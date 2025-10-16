import { useEffect, useState } from "react"

type AssistantState = "idle" | "listening" | "thinking" | "speaking"

interface AssistantCharacterProps {
  state: AssistantState
}

export default function AssistantCharacter({ state }: AssistantCharacterProps) {
  const [particles, setParticles] = useState<Array<{ id: number; delay: number }>>([])

  useEffect(() => {
    // Generate particles for visual effects
    const newParticles = Array.from({ length: 8 }, (_, i) => ({
      id: i,
      delay: i * 0.15,
    }))
    setParticles(newParticles)
  }, [])

  return (
    <div className="relative w-full max-w-sm aspect-square flex items-center justify-center">
      {/* Background Glow */}
      <div
        className={`absolute inset-0 rounded-full blur-3xl transition-all duration-700 ${
          state === "listening"
            ? "bg-primary/30 scale-110"
            : state === "thinking"
              ? "bg-accent/30 scale-105 animate-pulse"
              : state === "speaking"
                ? "bg-primary/40 scale-110"
                : "bg-primary/10 scale-100"
        }`}
      />

      {/* Outer Ring */}
      <div
        className={`absolute inset-8 rounded-full border-2 transition-all duration-500 ${
          state === "listening"
            ? "border-primary scale-110 animate-pulse-glow"
            : state === "thinking"
              ? "border-accent scale-105"
              : state === "speaking"
                ? "border-primary scale-110"
                : "border-primary/30 scale-100"
        }`}
      />

      {/* Sound Wave Bars (for listening/speaking) */}
      {(state === "listening" || state === "speaking") && (
        <div className="absolute inset-0 flex items-center justify-center gap-2">
          {particles.slice(0, 5).map((particle) => (
            <div
              key={particle.id}
              className="w-1.5 bg-primary rounded-full animate-wave"
              style={{
                height: "40%",
                animationDelay: `${particle.delay}s`,
              }}
            />
          ))}
        </div>
      )}

      {/* Main Character Circle */}
      <div
        className={`relative z-10 w-48 h-48 rounded-full flex items-center justify-center transition-all duration-500 ${
          state === "listening"
            ? "bg-primary/20 scale-110"
            : state === "thinking"
              ? "bg-accent/20 scale-105"
              : state === "speaking"
                ? "bg-primary/30 scale-110"
                : "bg-card scale-100"
        } border-2 ${state === "idle" ? "border-primary/30" : "border-primary"} shadow-2xl`}
      >
        {/* Character Face */}
        <div className="relative w-full h-full flex items-center justify-center">
          {/* Eyes */}
          <div className="absolute top-1/3 left-1/2 -translate-x-1/2 flex gap-8">
            <div
              className={`w-3 h-3 rounded-full bg-foreground transition-all duration-300 ${
                state === "thinking" ? "animate-pulse" : ""
              }`}
            />
            <div
              className={`w-3 h-3 rounded-full bg-foreground transition-all duration-300 ${
                state === "thinking" ? "animate-pulse" : ""
              }`}
            />
          </div>

          {/* Mouth */}
          <div className="absolute bottom-1/3 left-1/2 -translate-x-1/2">
            {state === "speaking" ? (
              <div className="w-12 h-8 border-2 border-foreground rounded-full border-t-0" />
            ) : state === "listening" ? (
              <div className="w-8 h-2 bg-foreground rounded-full" />
            ) : (
              <div className="w-10 h-1 bg-foreground/50 rounded-full" />
            )}
          </div>

          {/* Thinking Dots */}
          {state === "thinking" && (
            <div className="absolute bottom-12 left-1/2 -translate-x-1/2 flex gap-2">
              {[0, 1, 2].map((i) => (
                <div
                  key={i}
                  className="w-2 h-2 rounded-full bg-accent animate-bounce"
                  style={{ animationDelay: `${i * 0.15}s` }}
                />
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Floating Particles */}
      {state !== "idle" && (
        <div className="absolute inset-0">
          {particles.map((particle) => (
            <div
              key={particle.id}
              className="absolute w-2 h-2 rounded-full bg-primary/40 animate-float"
              style={{
                top: `${20 + particle.id * 10}%`,
                left: `${10 + particle.id * 10}%`,
                animationDelay: `${particle.delay}s`,
              }}
            />
          ))}
        </div>
      )}

      {/* State Label */}
      <div className="absolute -bottom-8 left-1/2 -translate-x-1/2 text-center">
        <p className="text-sm font-medium text-foreground capitalize">{state}</p>
      </div>
    </div>
  )
}