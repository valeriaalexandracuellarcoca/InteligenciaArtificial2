import cv2
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

# Directorios
input_dir = "Actas_Normalizadas"
output_dir = "Recortes_Actas"
os.makedirs(output_dir, exist_ok=True)

# Dimensiones de referencia (tu imagen de 2000x3057)
REF_ALTO = 2000
REF_ANCHO = 3057

# Definir coordenadas de las zonas de interés (basado en tus medidas)
ZONAS_INTERES = {
    'codigo_mesa': {
        'x1': 235, 'y1': 333, 'x2': 454, 'y2': 381,
        'nombre': 'codigo_mesa'
    },
    'votos_AP': {
        'x1': 1015, 'y1': 613, 'x2': 1174, 'y2': 667,
        'nombre': 'votos_AP'
    },
    'votos_LYP_ADN': {
        'x1': 1015, 'y1': 685, 'x2': 1174, 'y2': 736,
        'nombre': 'votos_LYP_ADN'
    },
    'votos_APB_SUMATE': {
        'x1': 1015, 'y1': 755, 'x2': 1174, 'y2': 806,
        'nombre': 'votos_APB_SUMATE'
    },
    'votos_LIBRE': {
        'x1': 1015, 'y1': 894, 'x2': 1174, 'y2': 946,
        'nombre': 'votos_LIBRE'
    },
    'votos_FP': {
        'x1': 1015, 'y1': 964, 'x2': 1174, 'y2': 1015,
        'nombre': 'votos_FP'
    },
    'votos_MAS_IPSP': {
        'x1': 1015, 'y1': 1033, 'x2': 1174, 'y2': 1086,
        'nombre': 'votos_MAS_IPSP'
    },
    'votos_UNIDAD': {
        'x1': 1015, 'y1': 1173, 'x2': 1174, 'y2': 1226,
        'nombre': 'votos_UNIDAD'
    },
    'votos_PDC': {
        'x1': 1015, 'y1': 1242, 'x2': 1174, 'y2': 1296,
        'nombre': 'votos_PDC'
    },
    'votos_validos': {
        'x1': 1015, 'y1': 1359, 'x2': 1174, 'y2': 1412,
        'nombre': 'votos_validos'
    },
    'votos_blancos': {
        'x1': 1015, 'y1': 1481, 'x2': 1174, 'y2': 1534,
        'nombre': 'votos_blancos'
    },
    'votos_nulos': {
        'x1': 1015, 'y1': 1551, 'x2': 1174, 'y2': 1603,
        'nombre': 'votos_nulos'
    }
}

def calcular_coordenadas_adaptativas(imagen, zonas):
    """
    Adapta las coordenadas al tamaño real de la imagen
    """
    h, w = imagen.shape[:2]
    factor_x = w / REF_ANCHO
    factor_y = h / REF_ALTO
    
    zonas_adaptadas = {}
    
    for nombre, zona in zonas.items():
        x1 = int(zona['x1'] * factor_x)
        y1 = int(zona['y1'] * factor_y)
        x2 = int(zona['x2'] * factor_x)
        y2 = int(zona['y2'] * factor_y)
        
        zonas_adaptadas[nombre] = {
            'coordenadas': (x1, y1, x2, y2),
            'nombre': zona['nombre']
        }
    
    return zonas_adaptadas

def recortar_zona(imagen, coordenadas):
    """
    Recorta una zona específica de la imagen
    """
    x1, y1, x2, y2 = coordenadas
    return imagen[y1:y2, x1:x2]

def procesar_actas():
    """
    Procesa todas las actas y guarda los recortes
    """
    archivos = glob.glob(os.path.join(input_dir, "*.jpg"))
    
    for i, path in enumerate(archivos):
        # Leer imagen
        img = cv2.imread(path, 0)  # Leer en escala de grises
        if img is None:
            print(f"No se pudo leer: {path}")
            continue
        
        # Calcular coordenadas adaptativas
        zonas_actuales = calcular_coordenadas_adaptativas(img, ZONAS_INTERES)
        
        # Crear carpeta para esta acta
        nombre_base = os.path.splitext(os.path.basename(path))[0]
        carpeta_acta = os.path.join(output_dir, nombre_base)
        os.makedirs(carpeta_acta, exist_ok=True)
        
        # Recortar y guardar cada zona
        for nombre, zona in zonas_actuales.items():
            recorte = recortar_zona(img, zona['coordenadas'])
            
            # Guardar recorte
            nombre_archivo = f"{zona['nombre']}.jpg"
            ruta_guardado = os.path.join(carpeta_acta, nombre_archivo)
            cv2.imwrite(ruta_guardado, recorte)
        
        if i % 50 == 0:  # Mostrar progreso cada 50 actas
            print(f"Procesadas {i}/{len(archivos)} actas")
    
    print(f"✅ Procesamiento completo. Recortes guardados en: {output_dir}")

# Función para visualizar los recortes de una acta de ejemplo
def mostrar_recortes_ejemplo():
    """
    Muestra los recortes de una acta para verificar
    """
    archivos = glob.glob(os.path.join(input_dir, "*.jpg"))
    if not archivos:
        print("No hay actas procesadas")
        return
    
    # Tomar primera acta
    img_path = archivos[0]
    img = cv2.imread(img_path, 0)
    
    # Calcular coordenadas
    zonas = calcular_coordenadas_adaptativas(img, ZONAS_INTERES)
    
    # Mostrar imagen original con rectángulos
    img_color = cv2.imread(img_path)
    img_con_rectangulos = img_color.copy()
    
    for nombre, zona in zonas.items():
        x1, y1, x2, y2 = zona['coordenadas']
        cv2.rectangle(img_con_rectangulos, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_con_rectangulos, nombre, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Redimensionar para visualización
    escala_visualizacion = 0.4
    h, w = img_con_rectangulos.shape[:2]
    nueva_w = int(w * escala_visualizacion)
    nueva_h = int(h * escala_visualizacion)
    img_visual = cv2.resize(img_con_rectangulos, (nueva_w, nueva_h))
    
    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(img_visual, cv2.COLOR_BGR2RGB))
    plt.title('Zonas de interés identificadas')
    plt.axis('off')
    plt.show()
    
    # Mostrar algunos recortes de ejemplo
    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    axes = axes.ravel()
    
    for j, (nombre, zona) in enumerate(list(zonas.items())[:12]):
        recorte = recortar_zona(img, zona['coordenadas'])
        axes[j].imshow(recorte, cmap='gray')
        axes[j].set_title(nombre)
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()

# Ejecutar procesamiento
print("Iniciando procesamiento de actas...")
procesar_actas()

# Mostrar ejemplo de recortes
print("\nMostrando ejemplo de recortes...")
mostrar_recortes_ejemplo()