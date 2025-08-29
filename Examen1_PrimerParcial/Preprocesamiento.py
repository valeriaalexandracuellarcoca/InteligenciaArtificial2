import cv2
import glob
import os

# Directorio de entrada (tus actas originales)
input_dir = "Actas"

# Directorio de salida (para guardar actas normalizadas)
output_dir = "Actas_Normalizadas"
os.makedirs(output_dir, exist_ok=True)

# Alto estándar al que vamos a llevar todas las actas
ALTO_ESTANDAR = 2000  

# Recorremos todas las actas
for path in glob.glob(os.path.join(input_dir, "*.jpg")):  # usa "*.png" si tus imágenes son png
    # Leer imagen
    img = cv2.imread(path)
    if img is None:
        print(f"No se pudo leer: {path}")
        continue

    # Obtener dimensiones originales
    h, w = img.shape[:2]

    # Calcular relación de escalado
    relacion = ALTO_ESTANDAR / h
    nuevo_ancho = int(w * relacion)

    # Redimensionar manteniendo proporción
    img_resized = cv2.resize(img, (nuevo_ancho, ALTO_ESTANDAR))

    # Guardar imagen normalizada en carpeta de salida
    nombre = os.path.basename(path)
    cv2.imwrite(os.path.join(output_dir, nombre), img_resized)

print("✅ Normalización terminada. Todas las actas están en:", output_dir)
