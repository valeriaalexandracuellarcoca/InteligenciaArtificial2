import fitz  # PyMuPDF
import pandas as pd
import re
import os
import traceback 

def procesar_kardex_a_csv(pdf_path, csv_path):
    """
    Convierte un Kardex universitario en formato PDF a un archivo CSV 
    con cálculos de carga académica y promedio semestral.
    
    Usa get_text("blocks") y los ordena para reconstruir las líneas del PDF
    y una expresión regular corregida para analizar los datos.
    """
    
    try:
        if not os.path.exists(pdf_path):
            print(f"Error: No se encontró el archivo PDF en la ruta: {pdf_path}")
            return

        # --- 1. Extracción de Texto del PDF (Lógica V5 - Correcta) ---
        full_text = ""
        with fitz.open(pdf_path) as doc:
            for page in doc:
                # Obtenemos los bloques de texto (type 0)
                text_blocks = [b for b in page.get_text("blocks") if b[6] == 0]
                
                # Los ordenamos por posición (primero arriba-abajo, luego izquierda-derecha)
                text_blocks.sort(key=lambda b: (b[1], b[0])) # b[1] es y0, b[0] es x0
                
                # --- LÓGICA CLAVE: RECONSTRUCCIÓN DE LÍNEAS ---
                page_lines = []
                current_line = []
                last_y0 = 0 

                for b in text_blocks:
                    # b es (x0, y0, x1, y1, "text...", block_no, block_type)
                    y0 = b[1]
                    text = b[4].strip() # Obtener el texto del bloque

                    if not text: # Omitir bloques vacíos
                        continue
                        
                    # Si el 'y' actual es muy diferente al 'y' anterior,
                    # significa que es una NUEVA línea.
                    if abs(y0 - last_y0) > 5 and current_line:
                        page_lines.append(" ".join(current_line))
                        current_line = [text]
                    else:
                        current_line.append(text)
                    
                    last_y0 = y0
                
                # Asegurarse de añadir la última línea
                if current_line:
                    page_lines.append(" ".join(current_line))
                
                full_text += "\n".join(page_lines) + "\n"
                
        # --- FIN DE LA LÓGICA DE EXTRACCIÓN ---

        # --- 2. Extracción de Datos Generales (Carnet Universitario) ---
        match_carnet = re.search(r"Carnet Universitario\s+([\d-]+)", full_text)
        if not match_carnet:
            print("Error: No se pudo encontrar el 'Carnet Universitario' en el PDF.")
            return
            
        d_estudiante = match_carnet.group(1)

        # --- 3. Extracción de Renglones de Materias (Regex V6 - Corregida) ---
        
        # --- MODIFICACIÓN CLAVE ---
        # Se corrigió el patrón para la 2da Instancia opcional (G7).
        # Ahora usa un grupo "no capturador" (?: \s+(\d{1,3}) )?
        # que busca explícitamente (espacio + número) y es opcional.
        # Esto elimina la ambigüedad que hacía fallar las filas sin 2da Instancia.
        row_pattern = re.compile(
            # G1: Gestión (e.g., "1/2021 NS")
            r"(\d/\d{4}\s*(?:NS|IN|V)?)"
            # G2: Plan (e.g., "1")
            r"\s+(\d+)"
            # G3: Curso (Semestre_Correspondiente) (e.g., "1")
            r"\s+(\d+)"
            # G4: Sigla (Código_Materia) (e.g., "FIS100")
            r"\s+([A-Z]{3}\d{3,})"
            # G5: Nombre Materia (No-Codicioso)
            r"\s+(.*?)" 
            # G6: Final (Nota_Final)
            r"\s+(\d{1,3})"
            # G7: 2da. Instancia (Opcional - Corregido)
            r"(?:\s+(\d{1,3}))?" # Busca un grupo opcional de (espacio + número)
            # G8: Estado
            r"\s+(Aprobado|Reprobado|Abandono)",
            re.MULTILINE
        )
        
        parsed_data = row_pattern.findall(full_text)

        if not parsed_data:
            print("Error: No se encontraron datos de materias con el patrón esperado.")
            # Si vuelve a fallar, descomenta las 3 líneas siguientes
            # print("--- TEXTO EXTRAÍDO PARA DEPURAR ---")
            # print(full_text)
            # print("--- FIN DEL TEXTO ---")
            return

        # --- 4. Procesamiento con Pandas ---
        
        df = pd.DataFrame(parsed_data, columns=[
            'Semestre', 'Plan', 'Semestre_Correspondiente', 'Código_Materia', 
            'Nombre_Materia_Temp', 'Nota_Final', 'Segunda_Instancia', 'Estado'
        ])

        # Limpieza de datos numéricos
        # .fillna(0) es crucial para las filas sin 2da Instancia
        df['Segunda_Instancia'] = pd.to_numeric(df['Segunda_Instancia'].fillna(0)).astype(int)
        df['Nota_Final'] = pd.to_numeric(df['Nota_Final']).astype(int)
        
        # --- 5. Cálculos Requeridos ---

        df['nota_efectiva'] = df[['Nota_Final', 'Segunda_Instancia']].max(axis=1)
        carga_academica = df.groupby('Semestre')['Código_Materia'].transform('count')
        df['Carga_Académica'] = carga_academica
        promedio_semestral = df.groupby('Semestre')['nota_efectiva'].transform('mean')
        df['Promedio_Semestral'] = promedio_semestral.round(2) 

        # --- 6. Formateo Final del CSV ---
        
        df['D_Estudiante'] = d_estudiante
        
        columnas_finales = [
            'D_Estudiante',
            'Semestre',
            'Código_Materia',
            'Semestre_Correspondiente',
            'Nota_Final',
            'Estado',
            'Segunda_Instancia',
            'Carga_Académica',
            'Promedio_Semestral'
        ]
        
        df_final = df[columnas_finales]

        # --- 7. Guardar el CSV ---
        df_final.to_csv(csv_path, sep=';', index=False, encoding='utf-8-sig')
        
        print(f"¡Éxito! Archivo CSV guardado en: {csv_path}")
        print(f"Se procesaron {len(df_final)} filas.") # ESTE NÚMERO AHORA DEBE SER CORRECTO

    # --- Bloque de reporte de errores ---
    except Exception as e:
        print("--- INICIO DEL REPORTE DE ERROR DETALLADO ---")
        print(f"Tipo de error: {type(e)}")
        print(f"Mensaje de error (repr): {repr(e)}")
        print("\n--- Traceback Completo (Rastreo del error) ---")
        traceback.print_exc()
        print("--- FIN DEL REPORTE DE ERROR ---")

# --- CÓMO USAR EL SCRIPT ---
pdf_file = "document.pdf"
csv_file = "kardex_procesado.csv"

if __name__ == "__main__":
    procesar_kardex_a_csv(pdf_file, csv_file)