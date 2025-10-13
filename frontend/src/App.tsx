// src/App.tsx
import React from "react";
import { Outlet } from "react-router-dom";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

export default function App() {
  const [status, setStatus] = React.useState<"ok" | "down" | "checking">("checking");

  React.useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(`${API_URL}/health`);
        if (!cancelled) setStatus(res.ok ? "ok" : "down");
      } catch {
        if (!cancelled) setStatus("down");
      }
    })();
    return () => { cancelled = true; };
  }, []);

  return (
    <div style={{ minHeight: "100vh", display: "grid", gridTemplateRows: "auto 1fr auto" }}>
      <header style={{ padding: "12px 16px", borderBottom: "1px solid #eee", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <h1 style={{ fontSize: 18, margin: 0 }}>Detector de Emoción</h1>
        <div style={{ fontSize: 12, color: status === "ok" ? "green" : "#b00" }}>
          {status === "checking" ? "checando API…" : status === "ok" ? "API OK" : "API caída"}
        </div>
      </header>

      <main style={{ padding: 16, maxWidth: 960, width: "100%", margin: "0 auto" }}>
        <Outlet />
      </main>

      <footer style={{ padding: 12, textAlign: "center", color: "#666", borderTop: "1px solid #eee" }}>
        Backend: <code>{API_URL}</code>
      </footer>
    </div>
  );
}
