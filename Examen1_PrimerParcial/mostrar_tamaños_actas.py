import cv2
import glob
import numpy as np

# Carpeta donde guardaste todas las actas (puedes cambiar la ruta y extensión)
ruta = "actas/*.jpg"   # o "actas/*.png" según tu caso

anchos = []
altos = []

# Recorre todas las imágenes
for archivo in glob.glob(ruta):
    img = cv2.imread(archivo)
    if img is not None:
        h, w = img.shape[:2]  # alto y ancho
        altos.append(h)
        anchos.append(w)

# Convertimos a arrays para facilidad de cálculo
altos = np.array(altos)
anchos = np.array(anchos)

# Estadísticas
print("---- ALTOS ----")
print("Promedio:", np.mean(altos))
print("Mínimo:", np.min(altos))
print("Máximo:", np.max(altos))

print("\n---- ANCHOS ----")
print("Promedio:", np.mean(anchos))
print("Mínimo:", np.min(anchos))
print("Máximo:", np.max(anchos))
