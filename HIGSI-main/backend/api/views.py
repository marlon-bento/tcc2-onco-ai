# api/views.py
from django.contrib.auth.models import User
from .serializers import UserSerializer, MyTokenObtainPairSerializer
from rest_framework import generics
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes
from rest_framework_simplejwt.views import TokenObtainPairView
from rest_framework.parsers import JSONParser, FormParser 


class UserCreateView(generics.CreateAPIView):
    queryset = User.objects.all()
    serializer_class = UserSerializer
    permission_classes = [AllowAny] 


class MyTokenObtainPairView(TokenObtainPairView):
    serializer_class = MyTokenObtainPairSerializer
    parser_classes = (JSONParser, FormParser)


@api_view(['GET'])
@permission_classes([IsAuthenticated]) 
def get_current_user(request):
    """
    Retorna o nome de usuário do usuário autenticado pelo token.
    Exatamente como o seu store espera: { "user": "username" }
    """
    return Response({"user": request.user.username})

from django.core.files.storage import FileSystemStorage
from .serializers import ExperimentoUploadSerializer
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework.permissions import IsAuthenticated
from rest_framework.views import APIView
import os
from django.conf import settings
from rest_framework import status

from skimage.segmentation import slic, mark_boundaries 
import numpy as np
import cv2
from api.preprocessar import (
    pre_processar_imagem,
    processar_e_gerar_grafo,
    get_superpixel_visualization,
    DEFAULT_MAX_DIM, 
    DEFAULT_N_NODES,
    DEFAULT_SEGMENTATION_TYPE
)
from api.predictor_brm import (
    main as predict_main,
)
MAX_DIM_PREPROCESS = DEFAULT_MAX_DIM    
SEGMENTATION_TYPE = DEFAULT_SEGMENTATION_TYPE 
TIPO_LESAO = 'CALC'
class UploadExperimentoView(APIView):
    
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated] 
    
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        
        serializer = ExperimentoUploadSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        image_file = serializer.validated_data['image']
        try:
            tipo_lesao_requisicao = request.data.get('tipo_lesao', TIPO_LESAO).upper()
            if tipo_lesao_requisicao not in ['CALC', 'MASS']:
                raise ValueError("Tipo de lesão inválido.")
        except Exception as e:
            pass
            
        try:
            try:
                n_superpixels_from_request = int(request.data.get('n_superpixels', DEFAULT_N_NODES))
                if n_superpixels_from_request not in [25, 50, 100, 200]:
                    n_superpixels_from_request = DEFAULT_N_NODES 
            except (ValueError, TypeError):
                n_superpixels_from_request = DEFAULT_N_NODES 


            
            base_name_prefix = "experimento"
            _base_name_orig, extension = os.path.splitext(image_file.name)
            
            original_storage_path = os.path.join(settings.MEDIA_ROOT, 'experimentos')
            original_storage = FileSystemStorage(location=original_storage_path, base_url=f'{settings.MEDIA_URL}experimentos/')

            nome_arquivo_base_atual = base_name_prefix 
            nome_arquivo_completo = f"{nome_arquivo_base_atual}{extension}"
            counter = 1
            
            while original_storage.exists(nome_arquivo_completo):
                nome_arquivo_base_atual = f"{base_name_prefix}_{counter}"
                nome_arquivo_completo = f"{nome_arquivo_base_atual}{extension}"
                counter += 1
            
            final_original_name_on_disk = original_storage.save(nome_arquivo_completo, image_file)
            absolute_original_image_url = request.build_absolute_uri(original_storage.url(final_original_name_on_disk))
            final_disk_path = original_storage.path(final_original_name_on_disk)


            img_bgr_original = cv2.imread(final_disk_path, cv2.IMREAD_COLOR)
            if img_bgr_original is None:
                raise ValueError("Não foi possível ler a imagem original para pré-processamento.")

            img_processed_clahe_rgb = pre_processar_imagem(img_bgr_original, max_dim=MAX_DIM_PREPROCESS)
            
            preprocess_storage_path = os.path.join(settings.MEDIA_ROOT, 'experimentos_preprocessados')
            os.makedirs(preprocess_storage_path, exist_ok=True)
            preprocess_storage = FileSystemStorage(location=preprocess_storage_path, base_url=f'{settings.MEDIA_URL}experimentos_preprocessados/')
            
            preprocess_file_name = f"imagem_preprocessada_{nome_arquivo_base_atual}{extension}"
            
            cv2.imwrite(preprocess_storage.path(preprocess_file_name), cv2.cvtColor(img_processed_clahe_rgb, cv2.COLOR_RGB2BGR))
            absolute_preprocess_image_url = request.build_absolute_uri(preprocess_storage.url(preprocess_file_name))


            print(f"Gerando grafo com {n_superpixels_from_request} nós...")
            graph_data = processar_e_gerar_grafo(
                img_processed_clahe_rgb, 
                n_nodes=n_superpixels_from_request, 
                segmentation_type=SEGMENTATION_TYPE,

            )
            print("Grafo gerado com sucesso.")

            print("Iniciando predição do modelo...")
            diagnosis_result = predict_main(
                graph_data, 
                nodes_to_test=n_superpixels_from_request, 
                max_dim_to_test=MAX_DIM_PREPROCESS,
                tipo_lesao=tipo_lesao_requisicao,
            ) 
            
            if diagnosis_result.get("error"):
                raise Exception(f"Erro na predição: {diagnosis_result['error']}")
            
            print("Predição concluída.")

            superpixel_viz_rgb = get_superpixel_visualization(
                img_processed_clahe_rgb, 
                n_nodes=n_superpixels_from_request, 
                segmentation_type=SEGMENTATION_TYPE
            )

            superpixel_storage_path = os.path.join(settings.MEDIA_ROOT, 'experimentos_superpixels')
            os.makedirs(superpixel_storage_path, exist_ok=True)
            superpixel_storage = FileSystemStorage(location=superpixel_storage_path, base_url=f'{settings.MEDIA_URL}experimentos_superpixels/')
            
            superpixel_file_name = f"imagem_com_superpixel_{nome_arquivo_base_atual}.png" 
            
            cv2.imwrite(superpixel_storage.path(superpixel_file_name), cv2.cvtColor(superpixel_viz_rgb, cv2.COLOR_RGB2BGR))
            absolute_superpixel_image_url = request.build_absolute_uri(superpixel_storage.url(superpixel_file_name))

            return Response({
                "message": "Processamento e diagnóstico concluídos.",
                "original_image_url": absolute_original_image_url,
                "preprocessada_image_url": absolute_preprocess_image_url,
                "superpixel_image_url": absolute_superpixel_image_url,
                "diagnosis": diagnosis_result 
            }, status=status.HTTP_201_CREATED)
        
        except Exception as e:
            import traceback
            traceback.print_exc() 
            return Response({"error": f"Erro interno ao processar a imagem: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)