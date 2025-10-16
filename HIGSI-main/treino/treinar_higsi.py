import os
import torch
import numpy as np
import mlflow
import mlflow.pytorch
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
from sklearn.preprocessing import StandardScaler

from prof.model import HIGSI

from decouple import config

# ==============================================================================
# --- PAINEL DE CONTROLE DO EXPERIMENTO ---
# ==============================================================================
# Variáveis de Configuração (Aqui você altera o cenário que quer rodar!)
# ----------------------------------------------------------------------
# Escolha o tipo de lesão: "MASS" ou "CALC"
TIPO_LESÃO = "MASS" 
# Escolha o tipo de segmentação/grafo: "SLIC" ou "DISF"
TIPO_SEGMENTAÇÃO = "SLIC" 

PASTA_FEATURES_MASS_SLIC = config("PASTA_FEATURES_MASS_HIGSI_SLIC")
PASTA_FEATURES_MASS_DISF = config("PASTA_FEATURES_MASS_HIGSI_DISF")
PASTA_FEATURES_CALC_SLIC = config("PASTA_FEATURES_CALC_HIGSI_SLIC")
PASTA_FEATURES_CALC_DISF = config("PASTA_FEATURES_CALC_HIGSI_DISF")

PASTA_FEATURES = ''
# --- Lógica de Seleção de Caminhos ---
if TIPO_LESÃO.upper() == "MASS":
    if TIPO_SEGMENTAÇÃO.upper() == "SLIC":
        PASTA_FEATURES = PASTA_FEATURES_MASS_SLIC
    elif TIPO_SEGMENTAÇÃO.upper() == "DISF":
        PASTA_FEATURES = PASTA_FEATURES_MASS_DISF
    else:
        raise ValueError("TIPO_SEGMENTAÇÃO deve ser 'SLIC' ou 'DISF'.")
elif TIPO_LESÃO.upper() == "CALC":
    if TIPO_SEGMENTAÇÃO.upper() == "SLIC":
        PASTA_FEATURES = PASTA_FEATURES_CALC_SLIC
    elif TIPO_SEGMENTAÇÃO.upper() == "DISF":
        PASTA_FEATURES = PASTA_FEATURES_CALC_DISF
    else:
        raise ValueError("TIPO_SEGMENTAÇÃO deve ser 'SLIC' ou 'DISF'.")
else:
    raise ValueError("TIPO_LESÃO deve ser 'MASS' ou 'CALC'.")

if not PASTA_FEATURES:
    raise ValueError("ERRO: Um dos caminhos essenciais não foi definido corretamente.")

# ==============================================================================
# --- PAINEL DE CONTROLE DO EXPERIMENTO ---
# ==============================================================================
MAX_DIM_A_TESTAR = 512
N_NODES_A_TESTAR = 50 

EPOCHS = 150
BATCH_SIZE = 16
LEARNING_RATE = 0.001
RANDOM_SEED = 42
TEST_SPLIT_SIZE = 0.13
VAL_SPLIT_SIZE = 0.13
NUM_CLASSES = 2
EMBEDDING_SIZE = 64
EARLY_STOPPING_PATIENCE = 35
WEIGHT_DECAY = 1e-3
# ==============================================================================

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_metrics(y_pred, y_true, epoch, run_type):
    acc = accuracy_score(y_true, y_pred)
    print(f"{run_type.capitalize()} Accuracy: {acc:.4f}")
    mlflow.log_metric(key=f"Accuracy-{run_type}", value=float(acc), step=epoch)
    return acc

def val(epoch, model, val_loader, loss_fn, device, run_type):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            pred = model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                batch_index=batch.batch,
                graph_index=batch.graph_index,
                reduced_index=batch.reduced_index
            )
            loss = loss_fn(pred, batch.y)
            total_loss += loss.item()
            all_preds.append(pred.argmax(dim=1).cpu().numpy())
            all_labels.append(batch.y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    acc = calculate_metrics(all_preds, all_labels, epoch, run_type)
    return total_loss / len(val_loader), acc

def train(epoch, model, train_loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    for batch in tqdm(train_loader, desc=f"Época {epoch+1}/{EPOCHS}", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            batch_index=batch.batch,
            graph_index=batch.graph_index,
            reduced_index=batch.reduced_index
        )
        loss = loss_fn(pred, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        all_preds.append(pred.argmax(dim=1).cpu().numpy())
        all_labels.append(batch.y.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    calculate_metrics(all_preds, all_labels, epoch, "train")
    return total_loss / len(train_loader)

def normalizar_features(lista_grafos):
    print("Normalizando as features dos nós...")
    all_node_features = torch.cat([data.x for data in lista_grafos], dim=0)
    scaler = StandardScaler()
    scaler.fit(all_node_features.numpy())
    for data in lista_grafos:
        data.x = torch.from_numpy(scaler.transform(data.x.numpy())).float()
    print("Normalização concluída.")
    return lista_grafos

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dim_folder_name = "original" if MAX_DIM_A_TESTAR is None else str(MAX_DIM_A_TESTAR)
    
    ARQUIVO_GRAFOS_PRONTOS = os.path.join(PASTA_FEATURES, dim_folder_name, f'{N_NODES_A_TESTAR}.pt')
    if not os.path.exists(ARQUIVO_GRAFOS_PRONTOS):
        raise FileNotFoundError(f"ERRO: Arquivo de grafos '{ARQUIVO_GRAFOS_PRONTOS}' não encontrado.")
    
    print(f"Carregando features de: {ARQUIVO_GRAFOS_PRONTOS}")
    lista_grafos = torch.load(ARQUIVO_GRAFOS_PRONTOS)
    lista_grafos = normalizar_features(lista_grafos)

    stratify_labels = [g.source for g in lista_grafos]
    train_val_data, test_data = train_test_split(
        lista_grafos, test_size=TEST_SPLIT_SIZE, stratify=stratify_labels, random_state=RANDOM_SEED
    )
    train_val_stratify_labels = [g.source for g in train_val_data]
    val_split_adjusted = VAL_SPLIT_SIZE / (1.0 - TEST_SPLIT_SIZE)
    train_data, val_data = train_test_split(
        train_val_data, test_size=val_split_adjusted, stratify=train_val_stratify_labels, random_state=RANDOM_SEED
    )

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    
    print("Dataset carregado e dividido:")
    print(f"  - Treino: {len(train_data)} | Validação: {len(val_data)} | Teste: {len(test_data)}")
    print(f"Usando dispositivo: {device}")

    model = HIGSI(
        node_feature_size=train_data[0].num_node_features,
        edge_feature_size=train_data[0].num_edge_features,
        num_classes=NUM_CLASSES,
        embedding_size=EMBEDDING_SIZE
    ).to(device)

    print(f"Modelo HIGSI carregado. Número de parâmetros: {count_parameters(model)}")
    
    save_dir = f"weights/HIGSI_{dim_folder_name}_{N_NODES_A_TESTAR}_seed{RANDOM_SEED}.pth"
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=15, min_lr=1e-6, factor=0.5)
    
    experiment_name = f"HIGSI - ImgSize {dim_folder_name} - Nodes {N_NODES_A_TESTAR}"
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(experiment_name)

    best_val_accu = 0
    patience_counter = 0

    with mlflow.start_run() as run:

        mlflow.log_param("model_type", "HIGSI")
        mlflow.log_param("epochs", EPOCHS)

        for epoch in range(EPOCHS):
            print(f"--- Época {epoch+1}/{EPOCHS} ---")
            train_loss = train(epoch, model, train_loader, optimizer, loss_fn, device)
            val_loss, val_accu = val(epoch, model, val_loader, loss_fn, device, "val")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            mlflow.log_metric(key="Loss-train", value=float(train_loss), step=epoch)
            mlflow.log_metric(key="Loss-val", value=float(val_loss), step=epoch)
            
            if val_accu > best_val_accu:
                best_val_accu = val_accu
                torch.save(model.state_dict(), save_dir)
                patience_counter = 0
                print(f"✨ Novo melhor modelo salvo com acurácia de validação: {best_val_accu:.4f}")
            else:
                patience_counter += 1
            
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"--- Parada antecipada na época {epoch+1} ---")
                break
            scheduler.step(val_accu)
    
    
        # ==============================================================================
        # --- INÍCIO DO BLOCO DE AVALIAÇÃO FINAL ---
        # ==============================================================================
        print("\n" + "="*60)
        print("Carregando o melhor modelo para avaliação final no conjunto de teste...")

        # Carrega os pesos do melhor modelo que foi salvo durante o treinamento
        model.load_state_dict(torch.load(save_dir))

        # Coloca o modelo em modo de avaliação (importante para desativar dropout, etc.)
        model.eval()

        # Listas para guardar as predições e os rótulos verdadeiros do conjunto de teste
        test_preds, test_labels = [], []

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                # Faz a predição com o melhor modelo
                pred = model(
                    x=batch.x,
                    edge_index=batch.edge_index,
                    edge_attr=batch.edge_attr,
                    batch_index=batch.batch,
                    graph_index=batch.graph_index,
                    reduced_index=batch.reduced_index
                )
                test_preds.extend(pred.argmax(dim=1).cpu().tolist())
                test_labels.extend(batch.y.cpu().tolist())

            # Exibe o relatório de classificação final com precisão, recall, f1-score, etc.
            print("\n--- Relatório de Classificação Final (Melhor Modelo no Teste) ---")
            final_accuracy = accuracy_score(test_labels, test_preds)
            print(f"Acurácia Final no Teste: {final_accuracy:.4f}\n")
            print(classification_report(test_labels, test_preds, target_names=["Benigno (0)", "Maligno (1)"]))

            # Loga as métricas finais no MLFlow
            final_metrics = classification_report(test_labels, test_preds, output_dict=True)
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