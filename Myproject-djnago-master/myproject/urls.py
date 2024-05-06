from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),    # URL pour l'interface d'administration Django
    path('', include('main.urls')),     # Inclure les URL de l'application principale
]
