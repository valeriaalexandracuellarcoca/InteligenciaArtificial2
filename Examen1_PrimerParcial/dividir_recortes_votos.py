import cv2
import os
import glob
from pathlib import Path

def crear_estructura_carpetas():
    """
    Crea la estructura de carpetas para guardar los recortes
    """
    # Carpeta principal para recortes de votos
    os.makedirs('Recortes_votos', exist_ok=True)
    print("✅ Carpeta Recortes_votos creada/verificada")

def dividir_imagen_votos(imagen_path, output_dir, categoria):
    """
    Divide una imagen de votos en 3 partes horizontales (centenas, decenas, unidades)
    y las guarda en la carpeta correspondiente
    """
    # Leer imagen
    img = cv2.imread(imagen_path, 0)  # Leer en escala de grises
    if img is None:
        print(f"❌ No se pudo leer: {imagen_path}")
        return
    
    # Obtener dimensiones
    h, w = img.shape
    
    # Calcular ancho de cada segmento
    segment_width = w // 3
    
    # Crear carpeta para esta categoría si no existe
    categoria_dir = os.path.join(output_dir, categoria)
    os.makedirs(categoria_dir, exist_ok=True)
    
    # Dividir y guardar los 3 dígitos
    for i, posicion in enumerate(['centenas', 'decenas', 'unidades']):
        x_start = i * segment_width
        x_end = (i + 1) * segment_width
        
        # Recortar dígito
        digito_img = img[:, x_start:x_end]
        
        # Guardar imagen del dígito
        nombre_digito = f"{categoria}_{posicion}.jpg"
        ruta_guardado = os.path.join(categoria_dir, nombre_digito)
        cv2.imwrite(ruta_guardado, digito_img)

def procesar_todas_actas():
    """
    Procesa todas las actas y divide las imágenes de votos en 3 dígitos
    """
    # Ruta de las actas recortadas
    actas_path = 'Recortes'
    
    # Verificar que existe la carpeta
    if not os.path.exists(actas_path):
        print(f"❌ No se encuentra la carpeta: {actas_path}")
        return
    
    # Listar todas las carpetas de actas
    carpetas_actas = [f for f in os.listdir(actas_path) 
                     if os.path.isdir(os.path.join(actas_path, f))]
    
    print(f"📁 Encontradas {len(carpetas_actas)} actas para procesar")
    
    # Categorías de votos a procesar
    categorias_votos = [
        'votos_AP', 'votos_LYP_ADN', 'votos_APB_SUMATE',
        'votos_LIBRE', 'votos_FP', 'votos_MAS_IPSP',
        'votos_UNIDAD', 'votos_PDC', 'votos_validos',
        'votos_blancos', 'votos_nulos'
    ]
    
    for i, carpeta_acta in enumerate(carpetas_actas):
        print(f"\n🔨 Procesando acta {i+1}/{len(carpetas_actas)}: {carpeta_acta}")
        
        # Ruta de la acta original
        acta_original_path = os.path.join(actas_path, carpeta_acta)
        
        # Ruta donde guardaremos los recortes de esta acta
        acta_output_path = os.path.join('Recortes_votos', carpeta_acta)
        os.makedirs(acta_output_path, exist_ok=True)
        
        # 1. Copiar imagen del código de mesa (no necesita división)
        codigo_path = os.path.join(acta_original_path, 'codigo_mesa.jpg')
        if os.path.exists(codigo_path):
            codigo_img = cv2.imread(codigo_path)
            cv2.imwrite(os.path.join(acta_output_path, 'codigo_mesa.jpg'), codigo_img)
        
        # 2. Procesar cada categoría de votos
        for categoria in categorias_votos:
            imagen_path = os.path.join(acta_original_path, f'{categoria}.jpg')
            
            if os.path.exists(imagen_path):
                # Dividir la imagen en 3 dígitos
                dividir_imagen_votos(imagen_path, acta_output_path, categoria)
            else:
                print(f"   ⚠️  No encontrado: {categoria}.jpg")
        
        if (i + 1) % 10 == 0:
            print(f"   ✅ {i+1} actas procesadas...")
    
    print(f"\n🎉 Procesamiento completado!")
    print("Estructura creada en: Recortes_votos/")
    print("Cada acta tiene:")
    print("  - codigo_mesa.jpg")
    print("  - carpetas para cada categoría con 3 dígitos cada una")

def mostrar_ejemplo():
    """
    Muestra un ejemplo de cómo quedaron los recortes
    """
    import matplotlib.pyplot as plt
    
    # Encontrar primera acta procesada
    actas_procesadas = [f for f in os.listdir('Recortes_votos') 
                       if os.path.isdir(os.path.join('Recortes_votos', f))]
    
    if not actas_procesadas:
        print("No hay actas procesadas para mostrar ejemplo")
        return
    
    acta_ejemplo = actas_procesadas[0]
    acta_path = os.path.join('Recortes_votos', acta_ejemplo)
    
    # Mostrar código de mesa
    codigo_path = os.path.join(acta_path, 'codigo_mesa.jpg')
    if os.path.exists(codigo_path):
        codigo_img = cv2.imread(codigo_path)
        plt.figure(figsize=(8, 4))
        plt.imshow(cv2.cvtColor(codigo_img, cv2.COLOR_BGR2RGB))
        plt.title('Código de Mesa')
        plt.axis('off')
        plt.show()
    
    # Mostrar ejemplo de dígitos divididos
    categorias = os.listdir(acta_path)
    categorias = [c for c in categorias if os.path.isdir(os.path.join(acta_path, c))]
    
    if categorias:
        categoria_ejemplo = categorias[0]
        digitos_path = os.path.join(acta_path, categoria_ejemplo)
        imagenes_digitos = sorted(os.listdir(digitos_path))
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for i, img_name in enumerate(imagenes_digitos):
            img_path = os.path.join(digitos_path, img_name)
            img = cv2.imread(img_path, 0)
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(img_name.split('.')[0])
            axes[i].axis('off')
        
        plt.suptitle(f'Dígitos de {categoria_ejemplo}')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Crear estructura de carpetas
    crear_estructura_carpetas()
    
    # Procesar todas las actas
    procesar_todas_actas()
    
    # Mostrar ejemplo
    mostrar_ejemplo()