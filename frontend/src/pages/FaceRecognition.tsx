// src/pages/FaceRecognition.tsx
import React from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

type FaceEmotion = {
  boundingPoly?: { x: number; y: number }[];
  scores?: { joy: number; sorrow: number; anger: number; surprise: number };
  top_emotion?: string;
  sentiment?: string;
};

export default function FaceRecognition() {
  const videoRef = React.useRef<HTMLVideoElement>(null);
  const canvasRef = React.useRef<HTMLCanvasElement>(null);

  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const [result, setResult] = React.useState<{ facesCount: number; faces: FaceEmotion[] } | null>(null);

  // Inicia cámara si no está activa
  async function ensureCamera() {
    if (!videoRef.current) return;
    if (videoRef.current.srcObject) return;
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    videoRef.current.srcObject = stream;
    await videoRef.current.play();
  }

  // Toma un frame, lo manda al backend y muestra emoción
  async function analyzeOnce() {
    setError(null);
    setBusy(true);
    try {
      await ensureCamera();
      const video = videoRef.current!;
      const canvas = canvasRef.current!;
      const ctx = canvas.getContext("2d")!;
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      const blob: Blob = await new Promise((res) => canvas.toBlob((b) => res(b as Blob), "image/jpeg", 0.9));
      const fd = new FormData();
      fd.append("file", blob, "frame.jpg");

      const resp = await fetch(`${API_URL}/emotion/analyze`, { method: "POST", body: fd });
      if (!resp.ok) {
        const txt = await resp.text();
        throw new Error(txt || `HTTP ${resp.status}`);
      }
      const data = await resp.json(); // { facesCount, faces: [...] }
      setResult(data);

      // Dibuja bounding boxes si vienen
      ctx.lineWidth = 2;
      ctx.strokeStyle = "#00ff00";
      if (data?.faces?.length) {
        data.faces.forEach((f: FaceEmotion) => {
          if (!f.boundingPoly || f.boundingPoly.length < 4) return;
          const pts = f.boundingPoly;
          ctx.beginPath();
          ctx.moveTo(pts[0].x, pts[0].y);
          for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
          ctx.closePath();
          ctx.stroke();
        });
      }
    } catch (e: any) {
      setError(e?.message || String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div style={{ display: "grid", gap: 16 }}>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        <video ref={videoRef} playsInline muted style={{ width: "100%", borderRadius: 12, background: "#000" }} />
        <canvas ref={canvasRef} style={{ width: "100%", borderRadius: 12, background: "#111" }} />
      </div>

      <button
        onClick={analyzeOnce}
        disabled={busy}
        style={{
          padding: "10px 14px",
          fontWeight: 600,
          borderRadius: 10,
          border: "1px solid #ccc",
          background: busy ? "#ddd" : "#0b5cff",
          color: busy ? "#666" : "#fff",
          cursor: busy ? "not-allowed" : "pointer",
          width: 220,
          justifySelf: "center",
        }}
      >
        {busy ? "Analizando…" : "Analizar emoción"}
      </button>

      {error && (
        <div style={{ color: "#b00", fontSize: 14, justifySelf: "center" }}>
          {error}
        </div>
      )}

      {result && (
        <div style={{ justifySelf: "center", textAlign: "center" }}>
          <div>Caras detectadas: <b>{result.facesCount}</b></div>
          {result.faces?.[0] && (
            <div style={{ marginTop: 8 }}>
              <div>Emoción dominante: <b>{result.faces[0].top_emotion ?? "—"}</b></div>
              {"sentiment" in result.faces[0] && (
                <div>Sentimiento: <b>{(result.faces[0] as any).sentiment}</b></div>
              )}
              {result.faces[0].scores && (
                <div style={{ marginTop: 6, fontSize: 14, color: "#444" }}>
                  joy: {result.faces[0].scores!.joy} ·
                  sorrow: {result.faces[0].scores!.sorrow} ·
                  anger: {result.faces[0].scores!.anger} ·
                  surprise: {result.faces[0].scores!.surprise}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
