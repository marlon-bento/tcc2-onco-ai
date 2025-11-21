# utils/image_processing.py

import torch
from PIL import Image, ImageDraw
import numpy as np
from torch_geometric.data import Data
from skimage.segmentation import slic, mark_boundaries # Para SLIC e visualização

# Se você tem o DISF, importe-o aqui.
# from your_disf_library import disf_segmentation 

# --- Funções de Pré-processamento ---

def preprocess_image_for_graph(pil_image: Image.Image, max_dim: int) -> np.ndarray:
    """
    Redimensiona e converte a imagem para o formato esperado pela segmentação.
    Geralmente, escala de cinza e um tamanho máximo.
    """
    original_width, original_height = pil_image.size
    
    if max(original_width, original_height) > max_dim:
        ratio = max_dim / max(original_width, original_height)
        new_size = (int(original_width * ratio), int(original_height * ratio))
        pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
    
    # Converte para escala de cinza e depois para array numpy
    image_np = np.array(pil_image.convert('L')) # 'L' para escala de cinza
    return image_np

def generate_graph_from_segments(image_np: np.ndarray, n_nodes_to_extract: int, 
                                 node_feature_size: int, edge_feature_size: int, 
                                 segmentation_type: str = "SLIC") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Aplica a segmentação (SLIC/DISF) e extrai features para o grafo.
    Retorna os tensores do grafo e os segmentos para visualização.

    Retorna:
        tuple: (node_features, edge_indices, edge_attrs, segments_labels)
    """
    print(f"[Image Processing] Gerando grafo com {segmentation_type} e {n_nodes_to_extract} nós.")

    if segmentation_type.upper() == "SLIC":
        # Ajuste compactness e sigma conforme seu treinamento
        segments_labels = slic(image_np, n_segments=n_nodes_to_extract, compactness=10, sigma=1, start_label=0)
    elif segmentation_type.upper() == "DISF":
        # Substitua por sua chamada DISF real
        # segments_labels = disf_segmentation(image_np, num_labels=n_nodes_to_extract)
        print("AVISO: DISF não implementado. Usando SLIC como fallback.")
        segments_labels = slic(image_np, n_segments=n_nodes_to_extract, compactness=10, sigma=1, start_label=0)
    else:
        raise ValueError(f"Tipo de segmentação '{segmentation_type}' desconhecido.")

    num_nodes = len(np.unique(segments_labels))

    # --- Lógica de extração de features de nós (EXEMPLO) ---
    # Esta é uma parte crucial e deve corresponder ao seu treinamento.
    # Exemplo: Média de pixel, desvio padrão, textura, etc., para cada superpixel.
    node_features_list = []
    for i in range(num_nodes):
        mask = (segments_labels == i)
        # Exemplo simples: média de intensidade de pixel
        if np.any(mask):
            mean_intensity = np.mean(image_np[mask])
            # Preencha com mais features se o seu modelo precisar
            node_feature = [mean_intensity] * node_feature_size # Preenche para o tamanho correto
            node_features_list.append(node_feature)
        else:
            # Caso um segmento esteja vazio (raro, mas possível com n_segments muito alto)
            node_features_list.append([0.0] * node_feature_size) 
            
    # Certifique-se de que node_features_list não está vazio
    if not node_features_list:
        raise ValueError("Nenhum nó foi extraído.")
        
    node_features = torch.tensor(node_features_list, dtype=torch.float32)

    # --- Lógica de construção de arestas (EXEMPLO) ---
    # Geralmente baseada na adjacência dos superpixels.
    # Um grafo de adjacência pode ser construído examinando os vizinhos de cada pixel.
    edges = []
    height, width = image_np.shape
    
    # Criar um mapeamento de vizinhança entre superpixels
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=bool)
    for y in range(height):
        for x in range(width):
            current_segment = segments_labels[y, x]
            # Vizinhos 4-conectados
            for dy, dx in [(0, 1), (1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    neighbor_segment = segments_labels[ny, nx]
                    if current_segment != neighbor_segment:
                        adj_matrix[current_segment, neighbor_segment] = True
                        adj_matrix[neighbor_segment, current_segment] = True

    edge_indices_list = []
    edge_attrs_list = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes): # Apenas uma vez por par (grafo não-direcionado)
            if adj_matrix[i, j]:
                edge_indices_list.append([i, j])
                edge_indices_list.append([j, i]) # Para grafo bidirecional (PyG espera isso)
                
                # Exemplo simples de feature de aresta: distância euclidiana entre centros de massa
                # ou diferença de cor média, etc.
                # Apenas um placeholder aqui
                edge_attrs_list.append([0.1] * edge_feature_size)
                edge_attrs_list.append([0.1] * edge_feature_size) # Para a aresta inversa

    if not edge_indices_list:
        # Se não houver arestas, criar um grafo com nós isolados
        # Ou levantar um erro, dependendo do que seu modelo espera
        print("AVISO: Nenhuma aresta foi gerada. Criando grafo com arestas artificiais ou vazio.")
        edge_indices = torch.empty((2, 0), dtype=torch.long)
        edge_attrs = torch.empty((0, edge_feature_size), dtype=torch.float32)
    else:
        edge_indices = torch.tensor(edge_indices_list, dtype=torch.long).t().contiguous()
        edge_attrs = torch.tensor(edge_attrs_list, dtype=torch.float32)
        
    # Validar dimensões
    if node_features.shape[1] != node_feature_size:
        raise ValueError(f"Tamanho de feature de nó inesperado. Esperado: {node_feature_size}, Obtido: {node_features.shape[1]}")
    if edge_attrs.shape[0] != edge_indices.shape[1] or edge_attrs.shape[1] != edge_feature_size:
        # Pode ocorrer se não houver arestas
        pass # A validação acima já lida com o caso de arestas vazias
        

    return node_features, edge_indices, edge_attrs, segments_labels

def visualize_superpixels(original_pil_image: Image.Image, segments_labels: np.ndarray) -> Image.Image:
    """
    Cria uma imagem PIL com os superpixels marcados sobre a imagem original.
    """
    # Certifique-se de que a imagem original está no modo correto para sobreposição
    if original_pil_image.mode != 'RGB':
        original_pil_image = original_pil_image.convert('RGB')
        
    image_np = np.array(original_pil_image)
    
    # mark_boundaries retorna uma imagem com as bordas marcadas
    # A cor da borda pode ser ajustada, o padrão é branco.
    segmented_image_np = mark_boundaries(image_np, segments_labels, color=(1, 0, 0)) # Bordas vermelhas
    
    # Converte de volta para PIL Image
    # mark_boundaries retorna float no range [0, 1], precisamos converter para [0, 255] uint8
    segmented_image_pil = Image.fromarray((segmented_image_np * 255).astype(np.uint8))
    
    return segmented_image_pil