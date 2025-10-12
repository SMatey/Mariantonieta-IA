import React, { useRef, useState } from "react";
import Webcam from "react-webcam";
import axios from "axios";
import { useNavigate, Link } from "react-router-dom";

const api = import.meta.env.VITE_API_URL as string;

export function LoginPage() {
  const webcamRef = useRef<Webcam>(null);
  const [userId, setUserId] = useState("");
  const [loading, setLoading] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const nav = useNavigate();

  const doLogin = async () => {
    setMsg(null);
    if (!userId.trim()) return setMsg("Ingresa user_id");
    const snap = webcamRef.current?.getScreenshot();
    if (!snap) return setMsg("No se pudo capturar la imagen.");
    const blob = await (await fetch(snap)).blob();

    const form = new FormData();
    form.append("user_id", userId.trim());
    form.append("file", new File([blob], "selfie.jpg", { type: "image/jpeg" }));

    setLoading(true);
    try {
      const { data } = await axios.post(`${api}/auth/login_multipart`, form);
      if (data.login === "success") {
        // Aquí podrías guardar un token/flag de sesión en localStorage
        nav("/app", { replace: true });   // -> página en blanco
      } else {
        setMsg(`Login: ${data.login}. Emoción: ${data?.emotion?.dominant ?? "n/a"}`);
      }
    } catch (e:any) {
      setMsg(e?.response?.data?.detail || e.message || "Error");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{minHeight:"100vh", display:"grid", placeItems:"center"}}>
      <div style={{width:420, display:"grid", gap:12}}>
        <h2>Iniciar sesión</h2>
        <input value={userId} onChange={e=>setUserId(e.target.value)} placeholder="user_id"
               style={{padding:"8px 12px", borderRadius:8, border:"1px solid #ccc"}} />
        <div style={{border:"1px solid #ddd", borderRadius:10, overflow:"hidden"}}>
          <Webcam ref={webcamRef} audio={false} screenshotFormat="image/jpeg"
                  videoConstraints={{width:640, height:480, facingMode:"user"}}
                  style={{width:"100%", height:"auto"}}/>
        </div>
        <button onClick={doLogin} disabled={loading}>{loading ? "Procesando..." : "Entrar"}</button>
        <div><Link to="/register">¿No tienes cuenta? Regístrate</Link></div>
        {msg && <div>{msg}</div>}
      </div>
    </div>
  );
}
