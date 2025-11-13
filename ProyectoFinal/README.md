# API de Predicción de Rendimiento Académico con TFT

Este proyecto implementa una API de Django REST Framework para predecir el promedio semestral de un estudiante utilizando un modelo pre-entrenado de *Temporal Fusion Transformer* (TFT).

La API carga el modelo y los artefactos de preprocesamiento al iniciar y expone un endpoint para recibir datos históricos y futuros, devolviendo una predicción cuantiliana del rendimiento.

## 1. Configuración del Entorno

### Prerrequisitos
- Python 3.8 o superior
- `pip` y `venv`

### Pasos de Instalación

1.  **Clonar el Repositorio (si aplica)**
    Si este proyecto estuviera en un repositorio git, lo clonarías. Por ahora, asume que ya tienes los archivos.

2.  **Crear y Activar un Entorno Virtual**
    Es una buena práctica aislar las dependencias del proyecto.

    ```bash
    # Crear el entorno virtual
    python -m venv venv

    # Activar en Windows
    .\venv\Scripts\activate

    # Activar en macOS/Linux
    source venv/bin/activate
    ```

3.  **Instalar Dependencias**
    El archivo `requirements.txt` contiene todas las librerías necesarias.

    ```bash
    pip install -r requirements.txt
    ```

4.  **Colocar los Artefactos del Modelo**
    Esta API requiere dos archivos que deben estar en la **carpeta raíz del proyecto** (al mismo nivel que `manage.py`):
    - `tft_model_weights.pth`: Los pesos del modelo PyTorch entrenado.
    - `tft_metadata.joblib`: Un archivo que contiene los `scalers` de normalización, la configuración del modelo y las listas de columnas (`historical_cols`, `future_cols`, `target_scaler`).

5.  **Iniciar el Servidor de Desarrollo**
    Una vez que todo está instalado, puedes correr el servidor de Django.

    ```bash
    # Navega a la carpeta que contiene manage.py
    cd tft_service

    # Inicia el servidor
    python manage.py runserver
    ```
    Si todo va bien, verás en la consola un mensaje indicando que el modelo y los artefactos se cargaron correctamente, y el servidor estará corriendo en `http://127.0.0.1:8000/`.

## 2. Cómo Usar la API

La API expone un único endpoint para realizar predicciones.

### Endpoint: `POST /api/predict/`

Este endpoint espera una petición `POST` con un cuerpo (`body`) en formato `JSON`.

#### Estructura del JSON de Entrada

El JSON debe contener dos claves principales:
- `kardex_csv_data` (string): Una cadena de texto que representa el historial académico del estudiante en formato CSV, **delimitado por punto y coma (`;`)**.
- `future_inputs` (objeto): Un objeto JSON con las variables conocidas del semestre futuro que se quiere predecir.

---


### Ejemplo de Uso con Postman

1.  **URL de la Petición:**
    - **Método:** `POST`
    - **URL:** `http://127.0.0.1:8000/api/predict/`

2.  **Cabeceras (Headers):**
    - `Content-Type`: `application/json`

3.  **Cuerpo (Body):**
    - Selecciona la opción `raw` y el formato `JSON`.
    - Pega el siguiente contenido:

    ```json
    {
        "kardex_csv_data": "ID_Estudiante;Semestre;Promedio_Semestral;Carga_Académica;Semestre_Correspondiente;Nota_Final;Estado;Segunda_Instancia\n12345;1/2022;85;5;1;85;A;0\n12345;1/2022;90;5;1;90;A;0\n12345;2/2022;75;4;2;75;A;0\n12345;2/2022;80;4;2;80;A;0\n12345;1/2023;60;5;3;60;R;1\n12345;1/2023;55;5;3;55;R;0",
        "future_inputs": {
            "Carga_Académica": 5,
            "Promedio_Semestre_Corresp": 4.0
        }
    }
    ```

    **Importante:**
    - El string de `kardex_csv_data` debe incluir los saltos de línea (`\n`) para separar las filas del CSV.
    - Las columnas en el CSV deben coincidir con las que el modelo espera.

---


### Respuesta de la API

Si la petición es exitosa, recibirás una respuesta `JSON` con el siguiente formato:

```json
{
    "prediccion": {
        "promedio_predicho": 68.5,
        "rango_inferior_q10": 62.1,
        "rango_superior_q90": 75.3,
        "estado_predicho": "✅ APROBADO",
        "riesgo": "🟡 MEDIO - Aprobación probable con esfuerzo"
    },
    "id_estudiante": 12345,
    "historial_promedio": 75.0
}
```

#### Descripción de la Respuesta:
- `promedio_predicho`: La predicción mediana (percentil 50) del promedio para el siguiente semestre.
- `rango_inferior_q10`: El percentil 10 de la predicción. Hay un 90% de probabilidad de que el promedio sea superior a este valor.
- `rango_superior_q90`: El percentil 90 de la predicción. Hay un 90% de probabilidad de que el promedio sea inferior a este valor.
- `estado_predicho`: "APROBADO" si el promedio predicho es 51 o más, "REPROBADO" en caso contrario.
- `riesgo`: Una evaluación cualitativa del riesgo de reprobación basada en el promedio predicho.
- `id_estudiante`: El identificador del estudiante, extraído del CSV.
- `historial_promedio`: El promedio simple de los semestres históricos del estudiante.

### Manejo de Errores

La API incluye validaciones para los datos de entrada y el estado del modelo:
- **`400 Bad Request`**: Si falta alguna clave en el JSON, si el CSV está vacío o si hay un error al procesar los datos.
- **`503 Service Unavailable`**: Si el modelo o los artefactos no se pudieron cargar al iniciar el servidor. Revisa los logs de la consola para más detalles.
- **`500 Internal Server Error`**: Si ocurre un error inesperado durante el preprocesamiento de los datos o la inferencia del modelo.
