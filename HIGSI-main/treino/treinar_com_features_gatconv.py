import os
import sys
import torch
import numpy as np
# para iniciar o servidor mlflow ui, rode no terminal:
# mlflow ui 
import mlflow
import mlflow.pytorch
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight


from prof.model import BRM_GATConv 

from decouple import config

PASTA_FEATURES = config("PASTA_FEATURES")
if not PASTA_FEATURES:
    raise ValueError("ERRO: Variável PASTA_FEATURES não encontrada. Verifique seu arquivo .env.")

# ==============================================================================
# --- PAINEL DE CONTROLE DO EXPERIMENTO ---
# ==============================================================================
MAX_DIM_A_TESTAR = 512
N_NODES_A_TESTAR = 200

# Parâmetros de Treinamento
EPOCHS = 150
BATCH_SIZE = 16
LEARNING_RATE = 0.001 
RANDOM_SEED = 42
TEST_SPLIT_SIZE = 0.10
VAL_SPLIT_SIZE = 0.10
NUM_CLASSES = 2

# Parâmetros do Modelo
EMBEDDING_SIZE = 200 
HEADS = 4 
DROPOUT_GNN = 0.3 
DROPOUT_CLASSIFIER = 0.3 
EARLY_STOPPING_PATIENCE = 30
WEIGHT_DECAY = 5e-4 
# ==============================================================================

# --- Garantir Reprodutibilidade ---
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)

def count_parameters(model):
    """Conta o número de parâmetros treináveis no modelo."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_metrics(y_pred, y_true, epoch, run_type):
    """Calcula e loga a acurácia."""
    acc = accuracy_score(y_true, y_pred)
    print(f"{run_type.capitalize()} Accuracy: {acc:.4f}")
    if mlflow.active_run():
      mlflow.log_metric(key=f"Accuracy-{run_type}", value=float(acc), step=epoch)
    return acc

def val(epoch, model, val_loader, loss_fn, device, run_type):
    """Executa uma época de validação ou teste."""
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
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
    all_preds, all_labels = [], []
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

def main():
    patience_counter = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dim_folder_name = "original" if MAX_DIM_A_TESTAR is None else str(MAX_DIM_A_TESTAR)
    ARQUIVO_GRAFOS_PRONTOS = os.path.join(PASTA_FEATURES, dim_folder_name, f'{N_NODES_A_TESTAR}.pt')
    if not os.path.exists(ARQUIVO_GRAFOS_PRONTOS):
        raise FileNotFoundError(f"ERRO: Arquivo de grafos '{ARQUIVO_GRAFOS_PRONTOS}' não encontrado.")
    print("="*60)
    print("Iniciando Experimento com MLFlow")
    print(f"  - Carregando features de: {ARQUIVO_GRAFOS_PRONTOS}")
    lista_grafos = torch.load(ARQUIVO_GRAFOS_PRONTOS, weights_only=False)
    from sklearn.preprocessing import StandardScaler
    print("Normalizando as features dos nós...")
    all_node_features = torch.cat([data.x for data in lista_grafos], dim=0)
    scaler = StandardScaler()
    scaler.fit(all_node_features.numpy())
    for data in lista_grafos:
        data.x = torch.from_numpy(scaler.transform(data.x.numpy())).float()
    print("Normalização concluída.")
    try:
        stratify_labels = [g.source for g in lista_grafos]
    except AttributeError:
        stratify_labels = [g.y.item() for g in lista_grafos]
    train_val_data, test_data = train_test_split(
        lista_grafos, test_size=TEST_SPLIT_SIZE, stratify=stratify_labels, random_state=RANDOM_SEED
    )
    train_val_stratify_labels = [g.source for g in train_val_data]
    val_split_adjusted = VAL_SPLIT_SIZE / (1.0 - TEST_SPLIT_SIZE)
    train_data, val_data = train_test_split(
        train_val_data, test_size=val_split_adjusted, stratify=train_val_stratify_labels, random_state=RANDOM_SEED
    )
    
    print("\n" + "-"*60)
    print("--- Configurando o WeightedRandomSampler para o Treino ---")
    train_sources = [g.source for g in train_data]
    source_counts = Counter(train_sources)
    print(f"Contagem das fontes no treino: {source_counts}")
    source_weights = {source: 1.0 / count for source, count in source_counts.items()}
    print(f"Pesos por fonte: {source_weights}")
    sample_weights = [source_weights[source] for source in train_sources]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, len(sample_weights))
    print("Sampler configurado.")
    print("-"*60)
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Dataset carregado e dividido:")
    print(f"  - Treino: {len(train_data)} grafos, Validação: {len(val_data)} grafos, Teste: {len(test_data)} grafos")
    print(f"Usando dispositivo: {device}")
    print("="*60)

    model = BRM_GATConv(
        node_feature_size=train_data[0].num_node_features,
        edge_feature_size=train_data[0].num_edge_features,
        num_classes=NUM_CLASSES,
        embedding_size=EMBEDDING_SIZE,
        heads=HEADS,
        dropout_gnn=DROPOUT_GNN,
        dropout_classifier=DROPOUT_CLASSIFIER
    ).to(device)

    print(f"Número de parâmetros do modelo (GATConv): {count_parameters(model)}")
    
    save_dir = f"weights/BRM_GATConv_{dim_folder_name}_{N_NODES_A_TESTAR}_seed{RANDOM_SEED}.pth"
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=15, min_lr=1e-6, factor=0.5)
    
    experiment_name = f"GATConv - ImgSize {dim_folder_name} - Nodes {N_NODES_A_TESTAR}" # Nome do experimento atualizado
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(experiment_name)

    best_val_accu = 0

    with mlflow.start_run() as run:
        mlflow.log_params({
            "model_type": "GATConv", "epochs": EPOCHS, "batch_size": BATCH_SIZE, 
            "learning_rate": LEARNING_RATE, "random_seed": RANDOM_SEED, 
            "image_dimension": dim_folder_name, "node_count": N_NODES_A_TESTAR,
            "embedding_size": EMBEDDING_SIZE, "attention_heads": HEADS,
            "weight_decay": WEIGHT_DECAY, "dropout_gnn": DROPOUT_GNN, 
            "dropout_classifier": DROPOUT_CLASSIFIER
        })

        for epoch in range(EPOCHS):
            print(f"--- Época {epoch+1}/{EPOCHS} ---")
            train_loss = train(epoch=epoch, model=model, train_loader=train_loader, optimizer=optimizer, loss_fn=loss_fn, device=device)
            print(f"Train Loss: {train_loss:.4f}")
            mlflow.log_metric(key="Loss-train", value=float(train_loss), step=epoch)
            val_loss, val_accu = val(epoch=epoch, model=model, val_loader=val_loader, loss_fn=loss_fn, device=device, run_type="val")
            print(f"Val Loss: {val_loss:.4f}")
            mlflow.log_metric(key="Loss-val", value=float(val_loss), step=epoch)
            _, _ = val(epoch=epoch, model=model, val_loader=test_loader, loss_fn=loss_fn, device=device, run_type="test")
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
    
    # --- AVALIAÇÃO FINAL DO MELHOR MODELO GATConv ---
    print("\n" + "="*60)
    print("Carregando o melhor modelo GATConv para avaliação final no conjunto de teste...")
    
    best_model = BRM_GATConv( 
        node_feature_size=test_data[0].num_node_features,
        edge_feature_size=test_data[0].num_edge_features,
        num_classes=NUM_CLASSES,
        embedding_size=EMBEDDING_SIZE,
        heads=HEADS,
        dropout_gnn=DROPOUT_GNN,
        dropout_classifier=DROPOUT_CLASSIFIER
    ).to(device)
    best_model.load_state_dict(torch.load(save_dir, weights_only=True))
    best_model.eval()

    test_preds, test_labels = [], []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = best_model(data.x, data.edge_index, data.edge_attr, data.batch)
            pred = out.argmax(dim=1)
            test_preds.extend(pred.cpu().tolist())
            test_labels.extend(data.y.cpu().tolist())
    print("\n--- Relatório de Classificação Final (Melhor Modelo no Teste) ---")
    final_accuracy = accuracy_score(test_labels, test_preds)
    print(f"Acurácia Final no Teste: {final_accuracy:.4f}\n")
    print(classification_report(test_labels, test_preds, target_names=["Benigno (0)", "Maligno (1)"]))
    final_metrics = classification_report(test_labels, test_preds, output_dict=True)
    if mlflow.active_run():
      mlflow.log_metric("final_test_accuracy", final_metrics['accuracy'])
      mlflow.log_metric("final_test_precision_benign", final_metrics['0']['precision'])
      mlflow.log_metric("final_test_recall_benign", final_metrics['0']['recall'])      
      mlflow.log_metric("final_test_f1_benign", final_metrics['0']['f1-score'])        
      mlflow.log_metric("final_test_precision_malign", final_metrics['1']['precision'])
      mlflow.log_metric("final_test_recall_malign", final_metrics['1']['recall'])      
      mlflow.log_metric("final_test_f1_malign", final_metrics['1']['f1-score'])
    print("="*60)

