// src/pages/Start.tsx
import React from "react";
import { useNavigate } from "react-router-dom";

export default function Home() {
  const navigate = useNavigate();
  return (
    <div style={{
      display: "grid",
      placeItems: "center",
      minHeight: "60vh",
      gap: 16,
      textAlign: "center"
    }}>
      <h2 style={{ margin: 0 }}>Bienvenid@</h2>
      <p style={{ color: "#555", marginTop: 4 }}>
        Presiona el botón para iniciar el reconocimiento facial y análisis de emoción.
      </p>
      <button
        onClick={() => navigate("/face")}
        style={{
          padding: "12px 18px",
          fontWeight: 700,
          borderRadius: 12,
          border: "1px solid #0b5cff",
          background: "#0b5cff",
          color: "#fff",
          cursor: "pointer",
          minWidth: 220
        }}
      >
        Face Recognition
      </button>
    </div>
  );
}
