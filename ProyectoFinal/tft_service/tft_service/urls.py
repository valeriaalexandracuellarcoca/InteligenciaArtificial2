from django.contrib import admin
from django.urls import path, include
from django.views.generic import TemplateView # <-- Importar esto

urlpatterns = [
    path('admin/', admin.site.urls),
    
    # Conecta las URLs de tu app 'predictor' bajo el prefijo 'api/'
    path('api/', include('predictor.urls')),
    
    # --- AÑADIR ESTA LÍNA ---
    # Sirve 'index.html' desde la raíz del proyecto
    path('', TemplateView.as_view(template_name='index.html'), name='home'),
]
