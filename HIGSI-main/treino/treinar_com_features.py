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

from prof.model import BRM

from decouple import config

PASTA_FEATURES = config("PASTA_FEATURES")
if not PASTA_FEATURES:
    raise ValueError("ERRO: Variável PASTA_FEATURES não encontrada. Verifique seu arquivo .env.")

# ==============================================================================
# --- PAINEL DE CONTROLE DO EXPERIMENTO ---
# ==============================================================================
# conjunto de features
MAX_DIM_A_TESTAR = 1024
N_NODES_A_TESTAR = 25  

# Parâmetros de Treinamento
EPOCHS = 150
BATCH_SIZE = 16
LEARNING_RATE = 0.001
RANDOM_SEED = 42
TEST_SPLIT_SIZE = 0.10 # % para teste
VAL_SPLIT_SIZE = 0.10   # % para validação (do que sobrar do treino)
NUM_CLASSES = 2
EMBEDDING_SIZE = 200 # Deve ser o mesmo usado no treino 

DROPOUT_GNN = 0.3
DROPOUT_CLASSIFIER = 0.3
EARLY_STOPPING_PATIENCE = 35 # Parar após x épocas sem melhora
# usar 1e-3, 5e-4, 2e-3
WEIGHT_DECAY = 5e-4 # regularização
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
            loss = loss_fn(pred, batch.y) # Adaptado para labels de uma dimensão
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
        loss = loss_fn(pred, batch.y) # Adaptado para labels de uma dimensão
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
    # ==============================================================================
    # --- SALVAR O MODELO TREINADO ---
    # ==============================================================================
    # 1. Pega o caminho absoluto do diretório onde este script está
    diretorio_do_script = os.path.dirname(os.path.abspath(__file__))

    # 2. Cria o caminho completo para o arquivo de pesos DENTRO de uma pasta 'weights'
    #    que ficará no mesmo diretório do script.
    save_dir = os.path.join(
        diretorio_do_script, 
        "weights", 
        f"BRM_{dim_folder_name}_{N_NODES_A_TESTAR}_seed{RANDOM_SEED}.pth"
    )

    # 3. Garante que o diretório 'weights' exista antes de salvar
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    return save_dir

def resgata_dados_treinamento(dim_folder_name):
    # Lógica para resgatar os dados de treinamento onde estão salvos
   
    ARQUIVO_GRAFOS_PRONTOS = os.path.join(PASTA_FEATURES, dim_folder_name, f'{N_NODES_A_TESTAR}.pt')
    if not os.path.exists(ARQUIVO_GRAFOS_PRONTOS):
        raise FileNotFoundError(f"ERRO: Arquivo de grafos '{ARQUIVO_GRAFOS_PRONTOS}' não encontrado.")
    print("="*60)
    print("Iniciando Experimento com MLFlow")
    print(f"  - Carregando features de: {ARQUIVO_GRAFOS_PRONTOS}")
    return torch.load(ARQUIVO_GRAFOS_PRONTOS, weights_only=False)

def main():
    patience_counter = 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- 2. CARREGAR E DIVIDIR OS DADOS ---
    dim_folder_name = "original" if MAX_DIM_A_TESTAR is None else str(MAX_DIM_A_TESTAR)

    lista_grafos = resgata_dados_treinamento(dim_folder_name)
    

    from sklearn.preprocessing import StandardScaler

    # --- NORMALIZAÇÃO DE FEATURES ---
    print("Normalizando as features dos nós...")
    # Juntar todas as features de todos os grafos em uma única matriz
    all_node_features = torch.cat([data.x for data in lista_grafos], dim=0)

    # Criar e treinar o normalizador APENAS nos dados
    scaler = StandardScaler()
    scaler.fit(all_node_features.numpy())

    # Aplicar a normalização a cada grafo individualmente
    for data in lista_grafos:
        data.x = torch.from_numpy(scaler.transform(data.x.numpy())).float()
    print("Normalização concluída.")
    # --- FIM DO NOVO BLOCO ---

    # Agora o código de divisão de dados continua 
    # labels = [g.y.item() for g in lista_grafos]
    print(lista_grafos[0])
    try:
        stratify_labels = [g.source for g in lista_grafos]
        print(f"Contagem de subgrupos para estratificação: {Counter(stratify_labels)}")
    except AttributeError:
        print("AVISO: Atributo '.source' não encontrado. Usando labels binários (0/1) para estratificação.")
        stratify_labels = [g.y.item() for g in lista_grafos]
    
    train_val_data, test_data = train_test_split(
        lista_grafos,
        test_size=TEST_SPLIT_SIZE,
        stratify=stratify_labels,
        random_state=RANDOM_SEED
    )

    
    # Dividir treino+validação em treino e validação
    # train_val_labels = [g.y.item() for g in train_val_data]
    # Ajusta o tamanho da validação para ser relativo ao conjunto de treino+validação
    train_val_stratify_labels = [g.source for g in train_val_data]
    val_split_adjusted = VAL_SPLIT_SIZE / (1.0 - TEST_SPLIT_SIZE)
    train_data, val_data = train_test_split(
        train_val_data,
        test_size=val_split_adjusted,
        stratify=train_val_stratify_labels,
        random_state=RANDOM_SEED
    )

    # printa os casos que existem no train data
    print(f"Contagem de classes no treino: {Counter([g.y.item() for g in train_data])}")
    print(f"Contagem de classes na validação: {Counter([g.y.item() for g in val_data])}")
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    # --- BLOCO DE VERIFICAÇÃO DA ESTRATIFICAÇÃO ---
    print("\n" + "-"*60)
    print("--- Verificando a Distribuição das Classes nos Conjuntos ---")

    # Contando as classes no conjunto de Treino
    train_labels = [g.source for g in train_data]
    print(f"Distribuição no Treino ({len(train_data)} amostras): {Counter(train_labels)}")

    # Contando as classes no conjunto de Validação
    val_labels = [g.source for g in val_data]
    print(f"Distribuição na Validação ({len(val_data)} amostras): {Counter(val_labels)}")

    # Contando as classes no conjunto de Teste
    test_labels = [g.source for g in test_data]
    print(f"Distribuição no Teste ({len(test_data)} amostras): {Counter(test_labels)}")

    print(f"Dataset carregado e dividido:")
    print(f"  - Treino: {len(train_data)} grafos")
    print(f"  - Validação: {len(val_data)} grafos")
    print(f"  - Teste: {len(test_data)} grafos")
    print(f"Usando dispositivo: {device}")
    print("="*60)

    # --- 3. CONFIGURAR MODELO, OTIMIZADOR E MLFLOW ---
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

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=15, min_lr=1e-6, factor=0.5)
    
    # Configurar MLFlow
    experiment_name = f"BRM - ImgSize {dim_folder_name} - Nodes {N_NODES_A_TESTAR}"
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(experiment_name)

    best_val_accu = 0

    # --- 4. EXECUTAR O LOOP DE TREINAMENTO ---
    with mlflow.start_run() as run:
        # Logar parâmetros no MLFlow
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("random_seed", RANDOM_SEED)
        mlflow.log_param("image_dimension", dim_folder_name)
        mlflow.log_param("node_count", N_NODES_A_TESTAR)

        for epoch in range(EPOCHS):
            print(f"--- Época {epoch+1}/{EPOCHS} ---")
            
            # Treino
            train_loss = train(epoch=epoch, model=model, train_loader=train_loader, optimizer=optimizer, loss_fn=loss_fn, device=device)
            # print(f"Train Loss: {train_loss:.4f}")
            mlflow.log_metric(key="Loss-train", value=float(train_loss), step=epoch)
            
            # Validação
            val_loss, val_accu = val(epoch=epoch, model=model, val_loader=val_loader, loss_fn=loss_fn, device=device, run_type="val")
            # print(f"Val Loss: {val_loss:.4f}")
            mlflow.log_metric(key="Loss-val", value=float(val_loss), step=epoch)
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            # Teste (opcional a cada época, bom para monitorar)
            # _, test_accu = val(epoch=epoch, model=model, val_loader=test_loader, loss_fn=loss_fn, device=device, run_type="test")
            
            # Salvar o melhor modelo
            if val_accu > best_val_accu:
                best_val_accu = val_accu
                torch.save(model.state_dict(), save_dir)
                mlflow.log_metric(key="Best Val Accuracy", value=float(best_val_accu), step=epoch)
                patience_counter = 0 # Reseta o contador
                print(f"✨ Novo melhor modelo salvo em '{save_dir}' com acurácia de validação: {best_val_accu:.4f}")
            else:
                patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"--- Parada antecipada na época {epoch+1} ---")
                break # Sai do loop de treino
            scheduler.step(val_accu)
        print("\n--- Treinamento Concluído ---")
    
    # --- 5. AVALIAÇÃO FINAL DO MELHOR MODELO ---
    print("\n" + "="*60)
    print("Carregando o melhor modelo para avaliação final no conjunto de teste...")
    
    # Instanciar uma nova arquitetura de modelo
    best_model = BRM(
        node_feature_size=test_data[0].num_node_features,
        edge_feature_size=test_data[0].num_edge_features,
        num_classes=NUM_CLASSES,
        embedding_size=EMBEDDING_SIZE
    ).to(device)

    # Carregar os pesos salvos
    best_model.load_state_dict(torch.load(save_dir, weights_only=True))
    best_model.eval()

    # Avaliar no conjunto de teste
    test_preds, test_labels = [], []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = best_model(data.x, data.edge_index, data.edge_attr, data.batch)
            pred = out.argmax(dim=1)
            test_preds.extend(pred.cpu().tolist())
            test_labels.extend(data.y.cpu().tolist())

    # Exibir o relatório de classificação final
    print("\n--- Relatório de Classificação Final (Melhor Modelo no Teste) ---")
    final_accuracy = accuracy_score(test_labels, test_preds)
    print(f"Acurácia Final no Teste: {final_accuracy:.4f}\n")
    print(classification_report(test_labels, test_preds, target_names=["Benigno (0)", "Maligno (1)"]))
    
    # Logar as métricas finais no MLFlow
    # PARA:
    # Logar as métricas finais no MLFlow
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
