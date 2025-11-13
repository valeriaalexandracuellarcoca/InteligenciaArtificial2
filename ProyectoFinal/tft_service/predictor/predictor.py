import torch
import pandas as pd
import numpy as np
import joblib
from .tft_model import TemporalFusionTransformer # Importar desde el archivo local

# Ubicación de los artefactos (asume que están en la carpeta raíz del proyecto Django)
MODEL_PATH = 'tft_model_weights.pth'
METADATA_PATH = 'tft_metadata.joblib'

def load_model_and_artifacts():
    """
    Carga el modelo, los pesos y los artefactos de preprocesamiento una sola vez.
    """
    try:
        # Cargar metadata (scalers, config, etc.)
        artifacts = joblib.load(METADATA_PATH)
        
        # Re-crear la arquitectura del modelo usando la config guardada
        model_config = artifacts['model_config']
        model = TemporalFusionTransformer(**model_config)
        
        # Cargar los pesos entrenados
        # Cargar en CPU, ya que el servidor de inferencia podría no tener GPU
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        
        # Poner el modelo en modo de evaluación (¡MUY IMPORTANTE!)
        model.eval()
        
        print("Modelo y artefactos cargados exitosamente.")
        return model, artifacts

    except FileNotFoundError:
        print(f"Error: No se encontraron los archivos del modelo. Asegúrate que '{MODEL_PATH}' y '{METADATA_PATH}' estén en la carpeta raíz.")
        return None, None
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return None, None

def make_prediction(model, artifacts, df_raw, future_inputs_dict):
    """
    Toma un DataFrame de kárdex en crudo y las pistas futuras,
    y devuelve una predicción.
    """
    
    # --- 1. Aplicar la misma LÓGICA DE AGREGACIÓN del entrenamiento ---
    try:
        df_raw['Semestre_Num'] = df_raw['Semestre'].str.split('/').apply(
            lambda x: int(x[1]) * 10 + int(x[0])
        )
        def tasa_aprobacion(x):
            if len(x) == 0: return 0
            return ((x == 'A') | (x == 'C')).sum() / len(x)
        def num_reprobadas(x):
            return ((x == 'R') | (x == 'B')).sum()
        agg_funcs = {
            'Promedio_Semestral': 'first', 'Carga_Académica': 'first',
            'Semestre_Correspondiente': 'mean', 'Nota_Final': ['mean', 'std'],
            'Estado': [tasa_aprobacion, num_reprobadas], 'Segunda_Instancia': 'sum'
        }
        df_agg = df_raw.groupby(['ID_Estudiante', 'Semestre', 'Semestre_Num']).agg(agg_funcs).reset_index()
        df_agg.columns = ['_'.join(col).strip('_') for col in df_agg.columns.values]
        df_agg = df_agg.rename(columns={
            'Promedio_Semestral_first': 'Promedio_Semestral',
            'Carga_Académica_first': 'Carga_Académica',
            'Semestre_Correspondiente_mean': 'Promedio_Semestre_Corresp',
            'Nota_Final_mean': 'Promedio_Notas', 'Nota_Final_std': 'Std_Notas',
            'Estado_tasa_aprobacion': 'Tasa_Aprobacion',
            'Estado_num_reprobadas': 'Num_Reprobadas',
            'Segunda_Instancia_sum': 'Num_Segunda_Instancia'
        })
        df_agg['Std_Notas'] = df_agg['Std_Notas'].fillna(0)
        df_agg = df_agg.sort_values('Semestre_Num')
        
        student_id = df_agg['ID_Estudiante'].iloc[0]
        promedio_historico = df_agg['Promedio_Semestral'].mean()
        
    except Exception as e:
        raise ValueError(f"Error al agregar los datos del CSV: {e}")

    # --- 2. Preparar Datos Históricos (X_hist) ---
    historical_cols = artifacts['historical_cols']
    historical_scaler = artifacts['historical_scaler']
    seq_len = artifacts['seq_len']

    # Tomar los últimos 'seq_len' semestres
    recent_data = df_agg.tail(seq_len).copy()
    recent_data[historical_cols] = historical_scaler.transform(recent_data[historical_cols])
    sequence_hist = recent_data[historical_cols].values

    # Aplicar padding si el historial es más corto que seq_len
    if len(sequence_hist) < seq_len:
        padding = np.zeros((seq_len - len(sequence_hist), sequence_hist.shape[1]))
        sequence_hist = np.vstack([padding, sequence_hist])
    
    X_hist_tensor = torch.FloatTensor(sequence_hist).unsqueeze(0) # [1, seq_len, num_hist_vars]

    # --- 3. Preparar Datos Futuros (X_future) ---
    future_cols = artifacts['future_cols']
    future_scaler = artifacts['future_scaler']

    future_df = pd.DataFrame([future_inputs_dict])
    future_df = future_df[future_cols] # Asegurar orden de columnas
    future_data_scaled = future_scaler.transform(future_df)
    
    X_future_tensor = torch.FloatTensor(future_data_scaled).unsqueeze(0) # [1, 1, num_future_vars]

    # --- 4. Hacer la Predicción ---
    with torch.no_grad():
        predictions_scaled, _ = model(X_hist_tensor, X_future_tensor)

    # --- 5. Desnormalizar y Formatear la Salida ---
    target_scaler = artifacts['target_scaler']
    predictions_np = predictions_scaled.cpu().numpy()
    predictions_denorm = target_scaler.inverse_transform(predictions_np.reshape(-1, 1)).reshape(predictions_np.shape)
    predictions_denorm = np.clip(predictions_denorm, 0, 100)

    pred_q10 = round(predictions_denorm[0, 0, 0], 1)
    pred_median = round(predictions_denorm[0, 0, 1], 1)
    pred_q90 = round(predictions_denorm[0, 0, 2], 1)

    if pred_median >= 70:
        riesgo = "BAJO - Excelente proyección"
    elif pred_median >= 51:
        riesgo = "MEDIO - Aprobación probable con esfuerzo"
    else:
        riesgo = "ALTO - Requiere intervención inmediata"
    
    resultado = {
        "promedio_predicho": pred_median,
        "rango_inferior_q10": pred_q10,
        "rango_superior_q90": pred_q90,
        "estado_predicho": "APROBADO" if pred_median >= 51 else "REPROBADO",
        "riesgo": riesgo
    }
    
    return resultado, student_id, round(promedio_historico, 2)
