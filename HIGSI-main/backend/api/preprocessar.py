import cv2
import numpy as np
import torch
from torch_geometric.data import Data
from skimage.segmentation import slic, mark_boundaries

from prof.graph_utils import RAG, RAG_DISF

DEFAULT_MAX_DIM = 1024
DEFAULT_N_NODES = 50
DEFAULT_SEGMENTATION_TYPE = "SLIC" # Ou "DISF"

def pre_processar_imagem(img_bgr_input: np.ndarray, max_dim: int = DEFAULT_MAX_DIM):
    img_processada = img_bgr_input.copy()
    
    if max_dim is not None:
        h, w = img_processada.shape[:2]
        if h > max_dim or w > max_dim:
            scale = max_dim / max(h, w)
            img_processada = cv2.resize(img_processada, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    
    img_rgb = cv2.cvtColor(img_processada, cv2.COLOR_BGR2RGB)
    # --- APLICAÇÃO DE CLAHE NA IMAGEM COLORIDA ---
    # a. Converte de RGB para o espaço de cor LAB
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    
    # b. Separa os canais (L=Luminosidade, a, b=Cor)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # c. Cria o objeto CLAHE e aplica APENAS no canal L
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_channel_clahe = clahe.apply(l_channel)
    
    # d. Junta os canais novamente
    lab_clahe = cv2.merge((l_channel_clahe, a_channel, b_channel))
    
    # e. Converte de volta para RGB
    img_rgb_final = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
    # -----------------------------------------------
    
    return img_rgb_final



def processar_e_gerar_grafo(img_rgb_clahe_processed: np.ndarray, 
                              n_nodes: int = DEFAULT_N_NODES, 
                              segmentation_type: str = DEFAULT_SEGMENTATION_TYPE) -> Data:
    
    img_processada = img_rgb_clahe_processed.copy()
    if img_processada is None:
        return None
    if segmentation_type.upper() == "SLIC":
        h_feat, edges, edge_feat, _= RAG(img_processada, n_nodes=n_nodes)
    elif segmentation_type.upper() == "DISF":
        h_feat, edges, edge_feat, _, _ = RAG_DISF(img_processada, n_nodes=n_nodes)
    else:
        raise ValueError("TIPO_SEGMENTAÇÃO deve ser 'SLIC' ou 'DISF'.")

    return Data(
        x=torch.tensor(h_feat, dtype=torch.float), 
        edge_index=torch.tensor(edges, dtype=torch.long),
        edge_attr=torch.tensor(edge_feat, dtype=torch.float), 
        y=torch.tensor([-1], dtype=torch.long),
        
        source="Uploaded Image", image_file_path="dynamic_upload"
    )


def get_superpixel_visualization(img_rgb_clahe_processed: np.ndarray, 
                                 n_nodes: int = DEFAULT_N_NODES,
                                 segmentation_type: str = DEFAULT_SEGMENTATION_TYPE) -> np.ndarray:
    """
    Gera uma imagem com a visualização dos superpixels sobre a imagem
    já pré-processada com CLAHE (esperada em RGB).
    Retorna a imagem de visualização em RGB.
    """
    if segmentation_type.upper() == "SLIC":
        segments_labels = slic(img_rgb_clahe_processed, n_segments=n_nodes, compactness=10, sigma=1, start_label=0)
    elif segmentation_type.upper() == "DISF":
        _, _, _, _, segments_labels = RAG_DISF(img_rgb_clahe_processed, n_nodes=n_nodes)
        
        if segments_labels is None:
            print("AVISO: DISF não forneceu labels de segmento para visualização. Usando SLIC como fallback para visualização.")
            segments_labels = slic(img_rgb_clahe_processed, n_segments=n_nodes, compactness=10, sigma=1, start_label=0)
    else:
        raise ValueError(f"Tipo de segmentação '{segmentation_type}' desconhecido. Use 'SLIC' ou 'DISF'.")

    img_with_superpixels_rgb = mark_boundaries(img_rgb_clahe_processed, segments_labels, color=(1, 0, 0)) 
    
    return (img_with_superpixels_rgb * 255).astype(np.uint8)