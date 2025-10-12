import os
import sys
import time
import torch
import numpy as np
from datetime import datetime, timedelta 
import locale
# --- Importações Específicas do Seu Projeto (Descomente se necessário) ---
# Se o NameError persistir, verifique o caminho 'prof.model'
from prof.model import BRM 
# from prof.grap_utils_new import RAG # Esta importação só é necessária no script de feature extraction

import cv2
import pandas as pd
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler 
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count 

from decouple import config
import mlflow
import mlflow.pytorch

# ==============================================================================
# --- PAINEL DE CONTROLE DO EXPERIMENTO ---
# ==============================================================================
# Variáveis do .env
# Certifique-se de que estas estão configuradas corretamente
ARQUIVO_MANIFEST = config("MANIFEST_FILE_PATH")
PASTA_IMAGENS = config("DATASET_IMAGES_PATH")
PASTA_FEATURES = config("PASTA_FEATURES")

if not ARQUIVO_MANIFEST or not PASTA_IMAGENS or not PASTA_FEATURES:
    # Esta verificação falha se o .env não estiver carregado, mas é mantida
    # para evitar um erro se a função main for chamada diretamente sem ambiente configurado.
    pass 

# conjunto de features
MAX_DIM_A_TESTAR = 1024
N_NODES_A_TESTAR = 25  

# Parâmetros de Treinamento
EPOCHS = 130
BATCH_SIZE = 16
LEARNING_RATE = 0.0005
RANDOM_SEED = 42
TEST_SPLIT_SIZE = 0.10 # % para teste
VAL_SPLIT_SIZE = 0.10   # % para validação (do que sobrar do treino)
NUM_CLASSES = 2
EMBEDDING_SIZE = 200 

# --- PARAMETRO DE BALANCEAMENTO ---
USE_WEIGHTED_LOSS = True 
# ----------------------------------

DROPOUT_GNN = 0.3
DROPOUT_CLASSIFIER = 0.3
EARLY_STOPPING_PATIENCE = 100 
WEIGHT_DECAY = 5e-4 
# ==============================================================================


# --- Garantir Reprodutibilidade ---
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)

# --- FUNÇÕES AUXILIARES ---

def count_parameters(model):
    """Conta o número de parâmetros treináveis no modelo."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_metrics(y_pred, y_true, epoch, run_type):
    """Calcula e loga a acurácia."""
    acc = accuracy_score(y_true, y_pred)
    print(f"{run_type.capitalize()} Accuracy: {acc:.4f}")
    mlflow.log_metric(key=f"Accuracy-{run_type}", value=float(acc), step=epoch)
    return acc

def val(epoch, model, val_loader, loss_fn, device, run_type):
    """Executa uma época de validação ou teste."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch) 
            loss = loss_fn(pred, batch.y) 
            total_loss += loss.item()
            
            all_preds.append(pred.argmax(dim=1).cpu().numpy())
            all_labels.append(batch.y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    acc = calculate_metrics(all_preds, all_labels, epoch, run_type)
    avg_loss = total_loss / len(val_loader)
    return avg_loss, acc

def train(epoch, model, train_loader, optimizer, loss_fn, device):
    """Executa uma época de treinamento."""

    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    for batch in tqdm(train_loader, desc=f"Época {epoch+1}/{EPOCHS}", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(x=batch.x, edge_index=batch.edge_index, edge_attr=batch.edge_attr, batch_index=batch.batch) 
        loss = loss_fn(pred, batch.y) 
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        all_preds.append(pred.argmax(dim=1).cpu().numpy())
        all_labels.append(batch.y.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    calculate_metrics(all_preds, all_labels, epoch, "train")
    avg_loss = total_loss / len(train_loader)
    return avg_loss

def salva_modelo_treinado(dim_folder_name):
    # --- SALVAR O MODELO TREINADO ---
    diretorio_do_script = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(
        diretorio_do_script, 
        "weights", 
        f"BRM_{dim_folder_name}_{N_NODES_A_TESTAR}_seed{RANDOM_SEED}.pth"
    )
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    return save_dir

def resgata_dados_treinamento(dim_folder_name):
    """Carrega os grafos de lesões (dataset completo)."""
    # Certifique-se de que PASTA_FEATURES está carregada corretamente
    PASTA_FEATURES = config("PASTA_FEATURES") 
    ARQUIVO_GRAFOS_PRONTOS = os.path.join(PASTA_FEATURES, dim_folder_name, f'{N_NODES_A_TESTAR}.pt')
    if not os.path.exists(ARQUIVO_GRAFOS_PRONTOS):
        raise FileNotFoundError(f"ERRO: Arquivo de grafos '{ARQUIVO_GRAFOS_PRONTOS}' não encontrado. Execute o script de extração de features primeiro.")
    print("="*60)
    print("Iniciando Experimento com MLFlow")
    print(f"  - Carregando features de: {ARQUIVO_GRAFOS_PRONTOS}")
    return torch.load(ARQUIVO_GRAFOS_PRONTOS, weights_only=False)

def resolver_conflitos_e_agregar_grafos(lista_grafos_lesoes):
    """
    Resolve o conflito de múltiplos diagnósticos por imagem e agrega os grafos,
    aplicando a regra do Máximo (MAX Pooling) na label.
    """
    print("\n" + "="*60)
    print("Resolvendo Conflitos: Agrupamento por Imagem (Max Pooling de Label)")
    print(f"  - Grafos de Lesão Carregados: {len(lista_grafos_lesoes)}")

    label_por_caminho = defaultdict(lambda: 0) 

    # 1. Encontrar a label final (MAX) para cada imagem
    for grafo in lista_grafos_lesoes:
        caminho = grafo.image_file_path
        lesao_label = grafo.y.item()
        label_por_caminho[caminho] = max(label_por_caminho[caminho], lesao_label)

    print(f"  - Total de Imagens Únicas: {len(label_por_caminho)}")

    # 2. Criar o novo dataset de grafos (um por imagem)
    primeiros_grafos = {}
    lista_grafos_unicos = []

    for grafo in lista_grafos_lesoes:
        caminho = grafo.image_file_path
        
        # Só consideramos o primeiro grafo encontrado para cada caminho de imagem
        if caminho not in primeiros_grafos:
            # Clona e atualiza a label para a label final da IMAGEM
            grafo_copia = grafo.clone()
            final_label = label_por_caminho[caminho]
            grafo_copia.y = torch.tensor([final_label], dtype=torch.long)
            
            primeiros_grafos[caminho] = True
            lista_grafos_unicos.append(grafo_copia)

    # 3. Verificar o resultado
    y_final = [g.y.item() for g in lista_grafos_unicos]
    print(f"  - Total de Grafos Únicos gerados: {len(lista_grafos_unicos)}")
    print(f"  - Contagem de Labels Finais: {Counter(y_final)}")
    print("="*60)
    
    return lista_grafos_unicos

def normalizar_features(lista_grafos):
    """Aplica a Normalização Z-score nas features de todos os nós."""
    print("Normalizando as features dos nós...")
    # Tenta concatenar features. Se x for uma lista vazia, ignora.
    try:
        all_node_features = torch.cat([data.x for data in lista_grafos], dim=0)
    except RuntimeError:
        print("AVISO: Lista de features vazia, pulando normalização.")
        return lista_grafos

    scaler = StandardScaler()
    scaler.fit(all_node_features.cpu().numpy()) # Treina na CPU para segurança

    for data in lista_grafos:
        # Aplica na CPU e volta para tensor, se necessário
        data.x = torch.from_numpy(scaler.transform(data.x.cpu().numpy())).float()
        # Não precisa mover de volta para o device aqui, pois será feito no DataLoader
        
    print("Normalização concluída.")
    return lista_grafos


def main():
    # printa os parâmetros do experimento
    print("="*60)
    print("Parâmetros do Experimento:")
    print(f"  - Dimensão da Imagem: {MAX_DIM_A_TESTAR if MAX_DIM_A_TESTAR else 'Original'}")
    print(f"  - Número de Nós: {N_NODES_A_TESTAR}")
    print(f"  - Épocas: {EPOCHS}")
    print(f"  - Batch Size: {BATCH_SIZE}")
    print(f"  - Learning Rate: {LEARNING_RATE}")
    print(f"  - Weight Decay: {WEIGHT_DECAY}")
    print(f"  - Embedding Size: {EMBEDDING_SIZE}")
    print(f"  - Dropout GNN: {DROPOUT_GNN} | Dropout Classifier: {DROPOUT_CLASSIFIER}")
    print(f"  - Test Split Size: {TEST_SPLIT_SIZE}")
    print(f"  - Val Split Size: {VAL_SPLIT_SIZE}")
    print(f"  - Random Seed: {RANDOM_SEED}")
    print(f"  - Early Stopping Patience: {EARLY_STOPPING_PATIENCE}")
    print(f"  - Usando Loss Ponderada: {USE_WEIGHTED_LOSS}")
    print("="*60)
    patience_counter = 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- 2. CARREGAR, RESOLVER CONFLITOS E DIVIDIR OS DADOS ---
    dim_folder_name = "original" if MAX_DIM_A_TESTAR is None else str(MAX_DIM_A_TESTAR)

    lista_grafos_lesoes = resgata_dados_treinamento(dim_folder_name)
    lista_grafos_unicos = resolver_conflitos_e_agregar_grafos(lista_grafos_lesoes)

    lista_grafos_unicos = normalizar_features(lista_grafos_unicos)


    # Usamos a lista_grafos_unicos para a divisão, estratificando pela label final (y)
    stratify_labels = [g.y.item() for g in lista_grafos_unicos] 
    
    # Divisão em Treino+Validação e Teste
    train_val_data, test_data = train_test_split(
        lista_grafos_unicos, 
        test_size=TEST_SPLIT_SIZE,
        stratify=stratify_labels,
        random_state=RANDOM_SEED
    )

    
    # Divisão em Treino e Validação
    train_val_stratify_labels = [g.y.item() for g in train_val_data]
    val_split_adjusted = VAL_SPLIT_SIZE / (1.0 - TEST_SPLIT_SIZE)
    train_data, val_data = train_test_split(
        train_val_data,
        test_size=val_split_adjusted,
        stratify=train_val_stratify_labels,
        random_state=RANDOM_SEED
    )

    # --- CÁLCULO DE PESOS (NOVO) ---
    train_labels_y = [g.y.item() for g in train_data] 
    class_counts = Counter(train_labels_y)
    total_samples = len(train_data)

    if USE_WEIGHTED_LOSS:
        weight_benign = total_samples / (NUM_CLASSES * class_counts[0])
        weight_malign = total_samples / (NUM_CLASSES * class_counts[1])
        
        class_weights = torch.tensor([weight_benign, weight_malign], dtype=torch.float32)
        print(f"\n--- Pesos da Perda Calculados ---")
        print(f"  - Count Benigno (0): {class_counts[0]}")
        print(f"  - Count Maligno (1): {class_counts[1]}")
        print(f"  - Pesos (B/M): {class_weights.tolist()}")
    else:
        class_weights = None
        print("\n--- Usando Perda Não Ponderada ---")
    # -------------------------------


    # Configuração dos Loaders
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    
    # --- BLOCO DE VERIFICAÇÃO DA ESTRATIFICAÇÃO ---
    print("\n" + "-"*60)
    print("--- Verificando a Distribuição das Classes nos Conjuntos ---")

    val_labels_y = [g.y.item() for g in val_data] 
    test_labels_y = [g.y.item() for g in test_data] 

    print(f"Distribuição no Treino ({len(train_data)} amostras): {Counter(train_labels_y)}")
    print(f"Distribuição na Validação ({len(val_data)} amostras): {Counter(val_labels_y)}")
    print(f"Distribuição no Teste ({len(test_data)} amostras): {Counter(test_labels_y)}")

    print(f"Dataset carregado e dividido:")
    print(f"  - Treino: {len(train_data)} grafos")
    print(f"  - Validação: {len(val_data)} grafos")
    print(f"  - Teste: {len(test_data)} grafos")
    print(f"Usando dispositivo: {device}")
    print("="*60)

    # --- 3. CONFIGURAR MODELO, OTIMIZADOR E MLFLOW ---
    # Move os pesos para o device
    if USE_WEIGHTED_LOSS:
        class_weights = class_weights.to(device)
        
    # NOTE: Certifique-se de que BRM está importado e acessível
    model = BRM( 
        node_feature_size=train_data[0].num_node_features,
        edge_feature_size=train_data[0].num_edge_features,
        num_classes=NUM_CLASSES,
        embedding_size=EMBEDDING_SIZE,
        dropout_gnn=DROPOUT_GNN,
        dropout_classifier=DROPOUT_CLASSIFIER
    ).to(device)

    print(f"Número de parâmetros do modelo: {count_parameters(model)}")
    
    save_dir = salva_modelo_treinado(dim_folder_name)

    # Inicializa a função de perda com os pesos
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=15, min_lr=1e-6, factor=0.5)
    
    # Configurar MLFlow
    experiment_name = f"BRM - ImgSize {dim_folder_name} - Nodes {N_NODES_A_TESTAR} - WLoss {USE_WEIGHTED_LOSS}"
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(experiment_name)

    best_val_accu = 0

    # --- 4. EXECUTAR O LOOP DE TREINAMENTO ---
    with mlflow.start_run() as run:
        # Logar parâmetros no MLFlow
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        # Log dos pesos
        if USE_WEIGHTED_LOSS:
            mlflow.log_param("loss_weight_benign", class_weights[0].item())
            mlflow.log_param("loss_weight_malign", class_weights[1].item())

        for epoch in range(EPOCHS):
            print(f"--- Época {epoch+1}/{EPOCHS} ---")
            
            train_loss = train(epoch=epoch, model=model, train_loader=train_loader, optimizer=optimizer, loss_fn=loss_fn, device=device)
            mlflow.log_metric(key="Loss-train", value=float(train_loss), step=epoch)
            
            val_loss, val_accu = val(epoch=epoch, model=model, val_loader=val_loader, loss_fn=loss_fn, device=device, run_type="val")
            mlflow.log_metric(key="Loss-val", value=float(val_loss), step=epoch)
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # Salvar o melhor modelo
            if val_accu > best_val_accu:
                best_val_accu = val_accu
                torch.save(model.state_dict(), save_dir)
                mlflow.log_metric(key="Best Val Accuracy", value=float(best_val_accu), step=epoch)
                patience_counter = 0 
                print(f"✨ Novo melhor modelo salvo em '{save_dir}' com acurácia de validação: {best_val_accu:.4f}")
            else:
                patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"--- Parada antecipada na época {epoch+1} ---")
                break 
            scheduler.step(val_accu)
        print("\n--- Treinamento Concluído ---")
    
    # --- 5. AVALIAÇÃO FINAL DO MELHOR MODELO ---
    print("\n" + "="*60)
    print("Carregando o melhor modelo para avaliação final no conjunto de teste...")
    
    best_model = BRM(
        node_feature_size=test_data[0].num_node_features,
        edge_feature_size=test_data[0].num_edge_features,
        num_classes=NUM_CLASSES,
        embedding_size=EMBEDDING_SIZE
    ).to(device)

    best_model.load_state_dict(torch.load(save_dir, weights_only=True))
    best_model.eval()

    test_preds, test_labels_final = [], []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = best_model(data.x, data.edge_index, data.edge_attr, data.batch)
            pred = out.argmax(dim=1)
            test_preds.extend(pred.cpu().tolist())
            test_labels_final.extend(data.y.cpu().tolist()) 

    # Exibir o relatório de classificação final
    print("\n--- Relatório de Classificação Final (Melhor Modelo no Teste) ---")
    final_accuracy = accuracy_score(test_labels_final, test_preds)
    print(f"Acurácia Final no Teste: {final_accuracy:.4f}\n")
    print(classification_report(test_labels_final, test_preds, target_names=["Benigno (0)", "Maligno (1)"]))
    
    # Logar as métricas finais no MLFlow
    final_metrics = classification_report(test_labels_final, test_preds, output_dict=True)
    mlflow.log_metric("final_test_accuracy", final_metrics['accuracy'])
    mlflow.log_metric("final_test_precision_benign", final_metrics['0']['precision'])
    mlflow.log_metric("final_test_recall_benign", final_metrics['0']['recall'])      
    mlflow.log_metric("final_test_f1_benign", final_metrics['0']['f1-score'])        
    mlflow.log_metric("final_test_precision_malign", final_metrics['1']['precision'])
    mlflow.log_metric("final_test_recall_malign", final_metrics['1']['recall'])      
    mlflow.log_metric("final_test_f1_malign", final_metrics['1']['f1-score'])        
    print("="*60)
    print("\n--- Treinamento Concluído ---")

if __name__ == "__main__":
    main()