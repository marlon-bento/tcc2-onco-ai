import os
import torch
import numpy as np

from prof.model import BRM


# ==============================================================================
# --- PAINEL DE CONTROLE DO EXPERIMENTO ---
# ==============================================================================
# Variáveis de Configuração (Aqui você altera o cenário que quer rodar!)
# ----------------------------------------------------------------------
# Escolha o tipo de lesão: "MASS" ou "CALC"
TIPO_LESÃO = "CALC" 
# Escolha o tipo de segmentação/grafo: "SLIC" ou "DISF"
TIPO_SEGMENTAÇÃO = "SLIC" 


# ==============================================================================
# --- PAINEL DE CONTROLE DO EXPERIMENTO ---
# ==============================================================================
# conjunto de features
MAX_DIM_A_TESTAR = 1024
N_NODES_A_TESTAR = 25  
NODE_FEATURE_SIZE = 103
EDGE_FEATURE_SIZE = 1


RANDOM_SEED = 42
NUM_CLASSES = 2
EMBEDDING_SIZE = 128 
DROPOUT_GNN = 0.3
DROPOUT_CLASSIFIER = 0.3


torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)

def count_parameters(model):
    """Conta o número de parâmetros treináveis no modelo."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def _get_model_path(dim_folder_name, tipo_segmentacao, tipo_lesao, n_nodes):
    """
    Retorna o caminho completo para o arquivo .pth do modelo treinado,
    assumindo que 'weights' está DENTRO de 'api'.
    """
    diretorio_do_script = os.path.dirname(os.path.abspath(__file__))
    
    model_path = os.path.join(
        diretorio_do_script, 
        "weights", 
        f"BRM_{dim_folder_name}_{n_nodes}_{tipo_segmentacao}_{tipo_lesao}.pth"
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ERRO: Arquivo do modelo (.pth) NÃO ENCONTRADO em {model_path}.")
    return model_path

def carrega_normalizador(dim_folder_name, tipo_segmentacao, tipo_lesao, n_nodes):
    import joblib
    from sklearn.preprocessing import StandardScaler
    diretorio_do_script = os.path.dirname(os.path.abspath(__file__))
    caminho_scaler = os.path.join(diretorio_do_script, "weights", f"scaler_{dim_folder_name}_{n_nodes}_{tipo_segmentacao}_{tipo_lesao}.joblib")
    if not os.path.exists(caminho_scaler):
        raise FileNotFoundError(f"Scaler não encontrado em {caminho_scaler}")
    scaler = joblib.load(caminho_scaler)
    return scaler
def normalizar_features(grafo_nao_normalizado, scaler):
    h_feat_np = grafo_nao_normalizado.x.numpy()
    h_feat_normalized = scaler.transform(h_feat_np)
    grafo_nao_normalizado.x = torch.tensor(h_feat_normalized, dtype=torch.float)
    return grafo_nao_normalizado

def main(graph_data_nao_normalizado, nodes_to_test=N_NODES_A_TESTAR, max_dim_to_test=MAX_DIM_A_TESTAR, tipo_lesao=TIPO_LESÃO, tipo_segmentacao=TIPO_SEGMENTAÇÃO):
    print(f"Iniciando predição com BRM para tipo de lesão: {tipo_lesao}, segmentação: {tipo_segmentacao}, nós: {nodes_to_test}, dimensão máxima: {max_dim_to_test}")
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dim_folder_name = "original" if max_dim_to_test is None else str(max_dim_to_test)
        
        scaler = carrega_normalizador(dim_folder_name, tipo_segmentacao, tipo_lesao, n_nodes=nodes_to_test)
        
        grafo_normalizado = normalizar_features(graph_data_nao_normalizado, scaler)
        

        print("\n" + "="*60)
        print("Carregando o melhor modelo para predição...")
        
        best_model = BRM(
            node_feature_size=NODE_FEATURE_SIZE,
            edge_feature_size=EDGE_FEATURE_SIZE,
            num_classes=NUM_CLASSES,
            embedding_size=EMBEDDING_SIZE,
            dropout_gnn=DROPOUT_GNN,        
            dropout_classifier=DROPOUT_CLASSIFIER 
        ).to(device)

        model_path = _get_model_path(dim_folder_name, tipo_segmentacao, tipo_lesao, n_nodes=nodes_to_test)
        best_model.load_state_dict(torch.load(model_path, map_location=device))
        best_model.eval() 

        print("Modelo carregado. Executando predição...")

        grafo_normalizado = grafo_normalizado.to(device)
        batch_index = torch.zeros(grafo_normalizado.num_nodes, dtype=torch.long).to(device)

        # Avaliar o grafo único
        with torch.no_grad():
            out_logits = best_model(
                x=grafo_normalizado.x, 
                edge_index=grafo_normalizado.edge_index, 
                edge_attr=grafo_normalizado.edge_attr, 
                batch_index=batch_index
            )

        # Processar o resultado
        probabilities = torch.softmax(out_logits, dim=1)
        predicted_class_idx = probabilities.argmax(dim=1).item()
        confidence = probabilities.max().item()
        prob_benign = probabilities[0][0].item()
        prob_malignant = probabilities[0][1].item()
        label = "Maligno" if predicted_class_idx == 1 else "Benigno"

        print(f"Predição concluída: {label} (Confiança: {confidence:.4f})")
        print("="*60)

        # Retornar o resultado
        return {
            "prediction": label,
            "predicted_class_id": predicted_class_idx,
            "confidence": f"{confidence:.4f}",
            "confidence_malignant": f"{prob_malignant:.4f}",
            "confidence_benign": f"{prob_benign:.4f}",
            "error": None
        }
    
    except FileNotFoundError as fnfe:
        print(f"ERRO: Arquivo essencial (modelo ou scaler) não encontrado: {fnfe}")
        return {"error": f"Erro de configuração do modelo: {fnfe}"}
    except Exception as e:
        print(f"ERRO inesperado durante a predição: {e}")
        import traceback
        traceback.print_exc() 
        return {"error": f"Falha na predição: {e}"}