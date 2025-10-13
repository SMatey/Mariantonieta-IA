// frontend/src/App.tsx
import { Link } from "react-router-dom";
export default function App() {
  return (
    <div style={{minHeight:"100vh", display:"grid", placeItems:"center"}}>
      <div style={{display:"grid", gap:12}}>
        <h1>Demo Face Auth</h1>
        <Link to="/register">Registrar</Link>
        <Link to="/login">Iniciar sesión</Link>
      </div>
    </div>
  );
}
