import React, { useRef, useState } from "react";
import Webcam from "react-webcam";
import axios from "axios";
import { useNavigate } from "react-router-dom";

const api = import.meta.env.VITE_API_URL as string;

export function RegisterPage() {
  const webcamRef = useRef<Webcam>(null);
  const [userId, setUserId] = useState("");
  const [shots, setShots] = useState<Blob[]>([]);
  const [loading, setLoading] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const nav = useNavigate();

  const takeShot = async () => {
    const snap = webcamRef.current?.getScreenshot();
    if (!snap) return;
    const blob = await (await fetch(snap)).blob();
    setShots(s => [...s, blob]);
  };

  const clearShots = () => setShots([]);

  const submit = async () => {
    if (!userId.trim()) return setMsg("Ingresa user_id");
    if (shots.length < 3) return setMsg("Toma al menos 3 fotos");

    const form = new FormData();
    form.append("user_id", userId.trim());
    shots.forEach((b, i) => form.append("files", new File([b], `ref_${i+1}.jpg`, { type: "image/jpeg" })));

    setLoading(true); setMsg(null);
    try {
      const { data } = await axios.post(`${api}/auth/enroll_user_multipart`, form);
      setMsg(`Registrado: ${data.stored_files} fotos`);
      // Opcional: ir a login automáticamente
      setTimeout(()=> nav("/login"), 800);
    } catch (e:any) {
      setMsg(e?.response?.data?.detail || e.message || "Error");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{minHeight:"100vh", display:"grid", placeItems:"center"}}>
      <div style={{width:460, display:"grid", gap:12}}>
        <h2>Registrar usuario</h2>
        <input value={userId} onChange={e=>setUserId(e.target.value)} placeholder="user_id"
               style={{padding:"8px 12px", borderRadius:8, border:"1px solid #ccc"}} />
        <div style={{border:"1px solid #ddd", borderRadius:10, overflow:"hidden"}}>
          <Webcam ref={webcamRef} audio={false} screenshotFormat="image/jpeg"
                  videoConstraints={{width:640, height:480, facingMode:"user"}}
                  style={{width:"100%", height:"auto"}}/>
        </div>

        <div style={{display:"flex", gap:8, flexWrap:"wrap"}}>
          <button onClick={takeShot} disabled={loading}>Tomar foto</button>
          <button onClick={clearShots} disabled={loading || shots.length===0}>Borrar</button>
          <button onClick={submit} disabled={loading || shots.length<3}>
            {loading ? "Enviando..." : "Registrar (3+ fotos)"}
          </button>
        </div>

        <div style={{display:"flex", gap:6, flexWrap:"wrap"}}>
          {shots.map((b, i) => (
            <img key={i} src={URL.createObjectURL(b)} style={{width:90, height:68, objectFit:"cover", borderRadius:6}}/>
          ))}
        </div>

        {msg && <div>{msg}</div>}
      </div>
    </div>
  );
}
