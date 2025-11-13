from django.urls import path
from .views import PredictView, PredictPDFView # <-- Importar la nueva vista

urlpatterns = [
    # Endpoint existente (para CSV en texto)
    path('predict/', PredictView.as_view(), name='predict'),
    
    # NUEVO Endpoint (para subir archivos PDF)
    path('predict-pdf/', PredictPDFView.as_view(), name='predict-pdf'),
]