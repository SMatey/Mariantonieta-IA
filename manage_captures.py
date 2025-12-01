#!/usr/bin/env python3
"""
Script para analizar y organizar las imágenes capturadas
"""

import os
from pathlib import Path
from collections import Counter
from datetime import datetime
import shutil

def analyze_captures():
    """Analiza las imágenes capturadas y muestra estadísticas"""
    
    captures_dir = Path("data/captures")
    
    if not captures_dir.exists():
        print("❌ La carpeta data/captures no existe")
        return
    
    # Buscar todas las imágenes
    image_files = list(captures_dir.glob("*.jpg"))
    
    if not image_files:
        print("ℹ️  No hay imágenes capturadas aún")
        return
    
    print(f"📊 ANÁLISIS DE {len(image_files)} IMÁGENES CAPTURADAS")
    print("=" * 50)
    
    # Extraer emociones de los nombres de archivo
    emotions = []
    dates = []
    
    for img_file in image_files:
        try:
            # Formato: YYYYMMDD_HHMMSS_mmm_EMOCION.jpg
            parts = img_file.stem.split("_")
            if len(parts) >= 4:
                emotion = parts[-1]  # Última parte es la emoción
                date_str = parts[0]  # Primera parte es la fecha
                emotions.append(emotion)
                dates.append(date_str)
        except:
            continue
    
    # Contar emociones
    emotion_counts = Counter(emotions)
    print("🎭 DISTRIBUCIÓN DE EMOCIONES:")
    for emotion, count in emotion_counts.most_common():
        percentage = (count / len(emotions)) * 100 if emotions else 0
        print(f"   {emotion}: {count} imágenes ({percentage:.1f}%)")
    
    # Contar por fechas
    date_counts = Counter(dates)
    print(f"\n📅 DISTRIBUCIÓN POR FECHA:")
    for date, count in sorted(date_counts.items(), reverse=True):
        try:
            formatted_date = datetime.strptime(date, "%Y%m%d").strftime("%d/%m/%Y")
        except:
            formatted_date = date
        print(f"   {formatted_date}: {count} imágenes")
    
    # Tamaño total
    total_size = sum(f.stat().st_size for f in image_files)
    total_mb = total_size / (1024 * 1024)
    print(f"\n💾 ESPACIO UTILIZADO: {total_mb:.1f} MB")

def organize_by_emotion():
    """Organiza las imágenes en subcarpetas por emoción"""
    
    captures_dir = Path("data/captures")
    
    if not captures_dir.exists():
        print("❌ La carpeta data/captures no existe")
        return
    
    image_files = list(captures_dir.glob("*.jpg"))
    
    if not image_files:
        print("ℹ️  No hay imágenes para organizar")
        return
    
    print("📁 ORGANIZANDO IMÁGENES POR EMOCIÓN")
    print("=" * 40)
    
    moved_count = 0
    
    for img_file in image_files:
        try:
            # Extraer emoción del nombre
            parts = img_file.stem.split("_")
            if len(parts) >= 4:
                emotion = parts[-1]
                
                # Crear carpeta de emoción si no existe
                emotion_dir = captures_dir / emotion
                emotion_dir.mkdir(exist_ok=True)
                
                # Mover imagen
                new_path = emotion_dir / img_file.name
                if not new_path.exists():
                    shutil.move(str(img_file), str(new_path))
                    moved_count += 1
                    print(f"   Movida: {img_file.name} → {emotion}/")
        except Exception as e:
            print(f"   ⚠️  Error con {img_file.name}: {e}")
    
    print(f"\n✅ {moved_count} imágenes organizadas")

def clean_old_captures(days_old=7):
    """Elimina imágenes más antiguas que X días"""
    
    captures_dir = Path("data/captures")
    
    if not captures_dir.exists():
        return
    
    from datetime import timedelta
    cutoff_date = datetime.now() - timedelta(days=days_old)
    
    deleted_count = 0
    
    # Buscar en carpeta principal y subcarpetas
    for img_file in captures_dir.rglob("*.jpg"):
        try:
            # Extraer fecha del nombre
            parts = img_file.stem.split("_")
            if len(parts) >= 2:
                date_str = parts[0]
                time_str = parts[1]
                
                # Parsear fecha y hora
                datetime_str = f"{date_str}_{time_str}"
                img_datetime = datetime.strptime(datetime_str, "%Y%m%d_%H%M%S")
                
                if img_datetime < cutoff_date:
                    img_file.unlink()
                    deleted_count += 1
                    print(f"   Eliminada: {img_file.name}")
        except:
            continue
    
    if deleted_count > 0:
        print(f"\n🗑️  {deleted_count} imágenes antiguas eliminadas")
    else:
        print("ℹ️  No hay imágenes antiguas para eliminar")

def main():
    print("🖼️  GESTOR DE IMÁGENES CAPTURADAS")
    print("=" * 40)
    print("1. Analizar imágenes capturadas")
    print("2. Organizar por emoción")
    print("3. Limpiar imágenes antigas (>7 días)")
    print("4. Todo lo anterior")
    
    choice = input("\nSelecciona una opción (1-4): ").strip()
    
    if choice == "1":
        analyze_captures()
    elif choice == "2":
        organize_by_emotion()
    elif choice == "3":
        clean_old_captures()
    elif choice == "4":
        analyze_captures()
        print("\n" + "="*50)
        organize_by_emotion()
        print("\n" + "="*50)
        clean_old_captures()
    else:
        print("❌ Opción inválida")

if __name__ == "__main__":
    main()