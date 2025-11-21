# api/urls.py
from django.urls import path
from . import views

urlpatterns = [
    # Rota para: POST /api/user/create
    path('user/create', views.UserCreateView.as_view(), name='user-create'),
    
    # Rota para: POST /api/user/login
    path('user/login', views.MyTokenObtainPairView.as_view(), name='user-login'),

    # Rota para: GET /api/user/me
    path('user/me', views.get_current_user, name='user-me'),

    # Rota para: POST /api/upload-experimento
    path('upload-experimento/', views.UploadExperimentoView.as_view(), name='upload-experimento'),
]