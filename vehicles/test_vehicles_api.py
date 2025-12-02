"""
Script de prueba para la API de detección de vehículos
Ejecutar después de colocar modelo_vehiculos.pkl en data/
"""

import requests
from pathlib import Path
import json

API_BASE = "http://localhost:8000"

def test_vehicles_health():
    """Prueba el endpoint de salud"""
    print("\n🔍 Probando /vehicles/health...")
    response = requests.get(f"{API_BASE}/vehicles/health")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    return response.status_code == 200

def test_vehicles_info():
    """Prueba el endpoint de información"""
    print("\n📋 Probando /vehicles/info...")
    response = requests.get(f"{API_BASE}/vehicles/info")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    return response.status_code == 200

def test_analyze_image(image_path: str):
    """Prueba el análisis de imagen"""
    print(f"\n🖼️  Probando /vehicles/analyze-image con {image_path}...")
    
    if not Path(image_path).exists():
        print(f"❌ Archivo no encontrado: {image_path}")
        return False
    
    with open(image_path, 'rb') as f:
        files = {'file': (Path(image_path).name, f, 'image/jpeg')}
        response = requests.post(f"{API_BASE}/vehicles/analyze-image", files=files)
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Vehículos detectados: {result['total_vehicles']}")
        print(f"   Conteo por tipo: {result['vehicle_counts']}")
        print(f"   Tiempo de procesamiento: {result['processing_time_ms']:.2f}ms")
        
        if result['detections']:
            print(f"\n   Primeras 3 detecciones:")
            for det in result['detections'][:3]:
                print(f"   - {det['class_name']}: {det['confidence']:.2%} confianza")
        
        if result['saved_image']:
            print(f"   Imagen guardada en: {result['saved_image']}")
        
        return True
    else:
        print(f"❌ Error: {response.text}")
        return False

def test_analyze_video(video_path: str, max_frames: int = 10):
    """Prueba el análisis de video"""
    print(f"\n🎬 Probando /vehicles/analyze-video con {video_path}...")
    
    if not Path(video_path).exists():
        print(f"❌ Archivo no encontrado: {video_path}")
        return False
    
    with open(video_path, 'rb') as f:
        files = {'file': (Path(video_path).name, f, 'video/mp4')}
        params = {'max_frames': max_frames}
        response = requests.post(
            f"{API_BASE}/vehicles/analyze-video",
            files=files,
            params=params
        )
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Video procesado")
        print(f"   Total frames: {result['total_frames']}")
        print(f"   Frames procesados: {result['processed_frames']}")
        print(f"   Promedio vehículos/frame: {result['average_vehicles_per_frame']:.2f}")
        print(f"   Total detecciones: {result['total_unique_detections']}")
        print(f"   Conteo por tipo: {result['vehicle_counts']}")
        print(f"   Tiempo de procesamiento: {result['processing_time_ms']:.2f}ms")
        return True
    else:
        print(f"❌ Error: {response.text}")
        return False

def test_general_health():
    """Prueba el health check general"""
    print("\n🏥 Probando /health general...")
    response = requests.get(f"{API_BASE}/health")
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Estado general: {result['overall_status']}")
        print(f"Vehicles disponible: {result.get('vehicles_available', False)}")
        
        if 'services' in result and 'vehicles' in result['services']:
            print(f"Estado del servicio vehicles: {result['services']['vehicles']}")
        
        return True
    else:
        print(f"❌ Error: {response.text}")
        return False

def main():
    """Ejecuta todas las pruebas"""
    print("=" * 60)
    print("🚗 PRUEBAS DE LA API DE DETECCIÓN DE VEHÍCULOS 🚗")
    print("=" * 60)
    
    # Verificar que el servidor esté corriendo
    try:
        response = requests.get(f"{API_BASE}/", timeout=5)
        print(f"✅ Servidor corriendo en {API_BASE}")
    except requests.exceptions.ConnectionError:
        print(f"❌ Error: Servidor no disponible en {API_BASE}")
        print("   Inicia el servidor con: python -m api.main")
        return
    
    # Ejecutar pruebas básicas
    results = []
    
    results.append(("Health Check General", test_general_health()))
    results.append(("Vehicles Health", test_vehicles_health()))
    results.append(("Vehicles Info", test_vehicles_info()))
    
    # Pruebas con archivos (opcional)
    print("\n" + "=" * 60)
    print("PRUEBAS CON ARCHIVOS (OPCIONAL)")
    print("=" * 60)
    print("\nPara probar con archivos reales:")
    print("1. Coloca una imagen en: test_image.jpg")
    print("2. Coloca un video en: test_video.mp4")
    print("3. Descomenta las líneas correspondientes en este script")
    
    # Descomenta estas líneas cuando tengas archivos de prueba:
    # results.append(("Analyze Image", test_analyze_image("test_image.jpg")))
    # results.append(("Analyze Video", test_analyze_video("test_video.mp4", max_frames=10)))
    
    # Resumen
    print("\n" + "=" * 60)
    print("RESUMEN DE PRUEBAS")
    print("=" * 60)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
    
    total = len(results)
    passed = sum(1 for _, r in results if r)
    print(f"\n{passed}/{total} pruebas exitosas")
    
    if passed == total:
        print("\n🎉 ¡Todas las pruebas pasaron!")
    else:
        print("\n⚠️  Algunas pruebas fallaron. Verifica los logs del servidor.")

if __name__ == "__main__":
    main()
