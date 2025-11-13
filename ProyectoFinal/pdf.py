import pdfplumber
import pandas as pd
import re
from pathlib import Path

def extraer_datos_kardex(pdf_path):
    """
    Extrae datos del kardex universitario desde un PDF usando detección de tablas
    """
    datos_materias = []
    carnet = "1"
    
    with pdfplumber.open(pdf_path) as pdf:
        # Extraer carnet del texto
        texto_primera_pagina = pdf.pages[0].extract_text()
        match_carnet = re.search(r'Carnet Universitario\s+(\d+-\d+)', texto_primera_pagina)
        if match_carnet:
            carnet = match_carnet.group(1).replace('-', '')
        
        # Procesar cada página
        for pagina in pdf.pages:
            # Extraer tablas de la página
            tablas = pagina.extract_tables()
            
            for tabla in tablas:
                for fila in tabla:
                    if not fila or len(fila) < 8:
                        continue
                    
                    # Identificar filas de datos (empiezan con gestión formato X/XXXX)
                    primera_col = str(fila[0]).strip() if fila[0] else ""
                    
                    # Buscar patrón de gestión (1/2021, 2/2021, etc.)
                    match_gestion = re.match(r'^(\d+/\d+)(IN|V|NS)?', primera_col)
                    
                    if match_gestion:
                        try:
                            gestion = match_gestion.group(1)
                            tipo_semestre = match_gestion.group(2) or ""
                            
                            # Ignorar semestres de verano (V)
                            if tipo_semestre == "V":
                                continue
                            
                            # Construir gestión completa (solo agregar IN, NS se omite)
                            gestion_completa = gestion + (tipo_semestre if tipo_semestre == "IN" else "")
                            
                            # Extraer curso (columna 2)
                            curso = str(fila[2]).strip() if len(fila) > 2 and fila[2] else ""
                            
                            # Extraer sigla (columna 3)
                            sigla = str(fila[3]).strip() if len(fila) > 3 and fila[3] else ""
                            
                            # Extraer nota final (penúltima o antepenúltima columna)
                            nota_final = None
                            segunda_instancia = None
                            estado = None
                            
                            # Buscar desde el final de la fila
                            for i in range(len(fila) - 1, -1, -1):
                                val = str(fila[i]).strip() if fila[i] else ""
                                
                                if val in ['Aprobado', 'Reprobado', 'Abandono', 'Convalidado']:
                                    estado = val
                                elif val.isdigit() or val == '-':
                                    if segunda_instancia is None:
                                        segunda_instancia = val
                                    elif nota_final is None:
                                        nota_final = val
                            
                            # Validar que tenemos los datos mínimos
                            if not all([curso, sigla, nota_final, estado]):
                                continue
                            
                            # Convertir estado: Aprobado=A, Reprobado=R, Convalidado=C, Abandono=B
                            if estado == 'Aprobado':
                                estado_codigo = 'A'
                            elif estado == 'Reprobado':
                                estado_codigo = 'R'
                            elif estado == 'Convalidado':
                                estado_codigo = 'C'
                            elif estado == 'Abandono':
                                estado_codigo = 'B'
                            else:
                                estado_codigo = 'R'  # Por defecto
                            
                            segunda_inst = 0 if segunda_instancia == '-' else int(segunda_instancia)
                            
                            datos_materias.append({
                                'gestion': gestion_completa,
                                'curso': int(curso),
                                'sigla': sigla,
                                'nota_final': int(nota_final),
                                'estado': estado_codigo,
                                'segunda_instancia': segunda_inst
                            })
                        
                        except (ValueError, IndexError) as e:
                            # Skip filas con errores de conversión
                            continue
    
    # Si no se encontraron materias con tablas, intentar con texto plano
    if not datos_materias:
        print("No se detectaron tablas, intentando extracción por texto...")
        datos_materias = extraer_por_texto(pdf_path, carnet)
    
    # Agrupar por semestre
    semestres = {}
    for materia in datos_materias:
        gestion = materia['gestion']
        if gestion not in semestres:
            semestres[gestion] = []
        semestres[gestion].append(materia)
    
    # Crear registros CSV
    registros_csv = []
    for gestion, materias in sorted(semestres.items()):
        carga_academica = len(materias)
        # Solo contar notas de materias Aprobadas (A) para el promedio
        notas_aprobadas = [m['nota_final'] for m in materias if m['estado'] == 'A']
        promedio = round(sum(notas_aprobadas) / len(notas_aprobadas), 1) if notas_aprobadas else 0.0
        
        for materia in materias:
            registros_csv.append({
                'D_Estudiante': carnet,
                'Semestre': gestion,
                'Código_Materia': materia['sigla'],
                'Semestre_Correspondiente': materia['curso'],
                'Nota_Final': materia['nota_final'],
                'Estado': materia['estado'],
                'Segunda_Instancia': materia['segunda_instancia'],
                'Carga_Académica': carga_academica,
                'Promedio_Semestral': promedio
            })
    
    return registros_csv, carnet

def extraer_por_texto(pdf_path, carnet):
    """
    Método alternativo: extrae datos línea por línea del texto
    """
    with pdfplumber.open(pdf_path) as pdf:
        texto_completo = ""
        for pagina in pdf.pages:
            texto_completo += pagina.extract_text() + "\n"
    
    datos_materias = []
    lineas = texto_completo.split('\n')
    
    for linea in lineas:
        # Patrón flexible que busca el formato del kardex
        patron = r'^(\d+/\d+)(IN|V|NS)?\s+\d+\s+(\d+)\s+([A-Z]{3}\d{3})\s+.*?\s+(\d+)\s+(-|\d+)\s+(Aprobado|Reprobado|Abandono)'
        match = re.search(patron, linea)
        
        if match:
            gestion = match.group(1)
            tipo_semestre = match.group(2) or "NS"
            
            # Ignorar semestres de verano (V)
            if tipo_semestre == "V":
                continue
            
            curso = match.group(3)
            sigla = match.group(4)
            nota_final = match.group(5)
            segunda_instancia = match.group(6)
            estado = match.group(7)
            
            gestion_completa = gestion if tipo_semestre == "NS" else f"{gestion}{tipo_semestre}"
            
            # Convertir estado: Aprobado=A, Reprobado=R, Convalidado=C, Abandono=B
            if estado == 'Aprobado':
                estado_codigo = 'A'
            elif estado == 'Reprobado':
                estado_codigo = 'R'
            elif estado == 'Convalidado':
                estado_codigo = 'C'
            elif estado == 'Abandono':
                estado_codigo = 'B'
            else:
                estado_codigo = 'R'
            
            segunda_inst = 0 if segunda_instancia == '-' else int(segunda_instancia)
            
            datos_materias.append({
                'gestion': gestion_completa,
                'curso': int(curso),
                'sigla': sigla,
                'nota_final': int(nota_final),
                'estado': estado_codigo,
                'segunda_instancia': segunda_inst
            })
    
    return datos_materias

def convertir_kardex_a_csv(pdf_path, csv_path=None):
    """
    Convierte un kardex PDF a formato CSV
    """
    if csv_path is None:
        csv_path = Path(pdf_path).with_suffix('.csv')
    
    print(f"Procesando {pdf_path}...")
    registros, carnet = extraer_datos_kardex(pdf_path)
    
    if not registros:
        print("❌ No se encontraron materias en el PDF")
        return None
    
    df = pd.DataFrame(registros)
    df.to_csv(csv_path, sep=';', index=False)
    
    print(f"✓ Archivo CSV generado: {csv_path}")
    print(f"✓ Total de registros: {len(registros)}")
    print(f"✓ Carnet: {carnet}")
    print(f"\nPrimeras 10 filas:")
    print(df.head(10).to_string(index=False))
    
    return df

if __name__ == "__main__":
    convertir_kardex_a_csv("document.pdf")