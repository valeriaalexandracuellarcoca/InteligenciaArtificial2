from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import pandas as pd
import io
import json

# Importar nuestra lógica de predicción
from .predictor import load_model_and_artifacts, make_prediction
# Importar el nuevo analizador de PDF
from .pdf_parser import clean_and_parse_kardex_pdf

# --- Cargar el modelo y los artefactos UNA SOLA VEZ al iniciar el servidor ---
print("Iniciando carga de modelo...")
MODEL, ARTIFACTS = load_model_and_artifacts()
print("Servidor listo para recibir peticiones.")
# -------------------------------------------------------------------------

class PredictView(APIView):
    """
    API View para recibir datos de kárdex como TEXTO CSV.
    """
    def post(self, request, *args, **kwargs):
        if MODEL is None or ARTIFACTS is None:
            return Response(
                {"error": "El modelo no está cargado."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )
        try:
            csv_data_string = request.data['kardex_csv_data']
            future_inputs = request.data['future_inputs']
        except KeyError:
            return Response(
                {"error": "El JSON debe contener 'kardex_csv_data' y 'future_inputs'."},
                status=status.HTTP_400_BAD_REQUEST
            )
        try:
            csv_file = io.StringIO(csv_data_string)
            df_raw = pd.read_csv(csv_file, sep=';')
            if df_raw.empty:
                raise ValueError("El CSV está vacío.")
        except Exception as e:
            return Response(
                {"error": f"Error al leer los datos del CSV: {e}"},
                status=status.HTTP_400_BAD_REQUEST
            )
        try:
            prediccion, student_id, hist_avg = make_prediction(
                MODEL, ARTIFACTS, df_raw, future_inputs
            )
        except Exception as e:
            return Response(
                {"error": f"Error durante el procesamiento: {e}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        return Response(
            {
                "prediccion": prediccion,
                "id_estudiante": student_id,
                "historial_promedio": hist_avg
            },
            status=status.HTTP_200_OK
        )

# --- NUEVO ENDPOINT PARA SUBIR PDFs ---
class PredictPDFView(APIView):
    """
    API View para recibir un archivo PDF de kárdex,
    procesarlo, y devolver una predicción.
    """
    
    def post(self, request, *args, **kwargs):
        
        # 1. Validar que el modelo esté cargado
        if MODEL is None or ARTIFACTS is None:
            return Response(
                {"error": "El modelo no está cargado. Revisa los logs del servidor."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )

        # 2. Obtener el archivo PDF y las pistas futuras
        try:
            pdf_file = request.FILES.get('kardex_pdf')
            future_inputs_json = request.POST.get('future_inputs_json')
            
            if not pdf_file:
                raise KeyError("No se encontró el archivo 'kardex_pdf'.")
            if not future_inputs_json:
                raise KeyError("No se encontró 'future_inputs_json'.")
                
            future_inputs = json.loads(future_inputs_json)

        except KeyError as e:
            return Response(
                {"error": f"Petición incompleta: {e}"},
                status=status.HTTP_400_BAD_REQUEST
            )
        except json.JSONDecodeError:
            return Response(
                {"error": "Formato de 'future_inputs_json' inválido."},
                status=status.HTTP_400_BAD_REQUEST
            )

        # 3. Procesar el PDF
        try:
            # Usar nuestra nueva función para convertir PDF -> DataFrame
            df_raw = clean_and_parse_kardex_pdf(pdf_file)
            
        except Exception as e:
            return Response(
                {"error": f"Error al procesar el PDF: {e}"},
                status=status.HTTP_400_BAD_REQUEST
            )

        # 4. Ejecutar la predicción (re-usamos la misma lógica)
        try:
            prediccion, student_id, hist_avg = make_prediction(
                MODEL, ARTIFACTS, df_raw, future_inputs
            )
        except Exception as e:
            return Response(
                {"error": f"Error durante la predicción: {e}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
            
        # 5. Devolver la respuesta exitosa
        return Response(
            {
                "prediccion": prediccion,
                "id_estudiante": student_id,
                "historial_promedio": hist_avg,
                "filas_procesadas": len(df_raw)
            },
            status=status.HTTP_200_OK
        )
