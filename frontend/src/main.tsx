// src/main.tsx
import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Routes, Route } from "react-router-dom";

import App from "./App";
import Home from "./pages/home";
import FaceRecognition from "./pages/FaceRecognition";

import "./index.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<App />}>
          <Route index element={<Home />} />     {/* <- pantalla con 1 botón */}
          <Route path="face" element={<FaceRecognition />} /> {/* <- tu flujo actual */}
        </Route>
      </Routes>
    </BrowserRouter>
  </React.StrictMode>
);
