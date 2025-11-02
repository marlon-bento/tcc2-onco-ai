import os
import sys
import torch
import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter

from prof.model import BRM


from decouple import config

# ==============================================================================
# --- FUNÇÕES AUXILIARES DE TREINAMENTO E DADOS ---
# ==============================================================================
def count_parameters(model):
    """Conta o número de parâmetros treináveis no modelo."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_metrics(y_pred, y_true, epoch, run_type):
    """Calcula e loga a acurácia."""
    acc = accuracy_score(y_true, y_pred)
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
    for batch in tqdm(train_loader, desc=f"Época {epoch+1}", leave=False):
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

def salva_modelo_treinado(dim_folder_name, tipo_lesao, n_nodes, run_id):
    """Salva o modelo com um nome único para o experimento."""
    diretorio_base = os.path.dirname(os.path.abspath(sys.argv[0])) if hasattr(sys, 'argv') and os.path.exists(os.path.dirname(os.path.abspath(sys.argv[0]))) else os.getcwd()
    save_dir = os.path.join(
        diretorio_base, 
        "weights", 
        f"{tipo_lesao}_BRM_{dim_folder_name}_{n_nodes}_RUN{run_id}.pth"
    )
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    return save_dir

def normalizar_features(lista_grafos):
    from sklearn.preprocessing import StandardScaler
    all_node_features = torch.cat([data.x for data in lista_grafos], dim=0)
    scaler = StandardScaler()
    scaler.fit(all_node_features.numpy())
    for data in lista_grafos:
        data.x = torch.from_numpy(scaler.transform(data.x.numpy())).float()
    return lista_grafos
    
def resgata_dados_treinamento(pasta_features, n_nodes, dim_folder_name):
    """Carrega os dados."""
    ARQUIVO_GRAFOS_PRONTOS = os.path.join(pasta_features, dim_folder_name, f'{n_nodes}.pt')
    if not os.path.exists(ARQUIVO_GRAFOS_PRONTOS):
        raise FileNotFoundError(f"ERRO: Arquivo de grafos '{ARQUIVO_GRAFOS_PRONTOS}' não encontrado.")
    
    return torch.load(ARQUIVO_GRAFOS_PRONTOS, weights_only=False)

# ==============================================================================
# --- DEFINIÇÃO DO ESPAÇO DE BUSCA E LOOP SEQUENCIAL ---
# ==============================================================================

# Define os valores que serão testados
param_grid = {
    'TIPO_LESÃO': ["CALC", "MASS"],
    'TIPO_SEGMENTAÇÃO': ["SLIC", "DISF"], 
    'MAX_DIM_A_TESTAR': [1024],
    'N_NODES_A_TESTAR': [25],
    'LEARNING_RATE': [1e-3, 5e-4],
    'WEIGHT_DECAY': [5e-4, 1e-3],
    'EMBEDDING_SIZE': [128],
    'DROPOUT_GNN': [0.3],
    'DROPOUT_CLASSIFIER': [0.3],
    'EPOCHS': [90], 
    'BATCH_SIZE': [16],
}

# ----------------------------------------------------------------------
# Lógica de Mapeamento de Caminhos 
# ----------------------------------------------------------------------
PASTA_FEATURES_MASS_SLIC = config("PASTA_FEATURES_MASS_SLIC")
PASTA_FEATURES_MASS_DISF = config("PASTA_FEATURES_MASS_DISF")
PASTA_FEATURES_CALC_SLIC = config("PASTA_FEATURES_CALC_SLIC")
PASTA_FEATURES_CALC_DISF = config("PASTA_FEATURES_CALC_DISF")




# ==============================================================================
# --- FUNÇÃO PRINCIPAL DE EXECUÇÃO DE EXPERIMENTO EM PARALELO ---
# ==============================================================================
def run_experiment_sequential(params): 
    """
    Roda um experimento completo para um conjunto de parâmetros. 
    """
    # 1. Definição de Parâmetros
    EPOCHS = params['EPOCHS']
    BATCH_SIZE = params['BATCH_SIZE']
    LEARNING_RATE = params['LEARNING_RATE']
    WEIGHT_DECAY = params['WEIGHT_DECAY']
    EMBEDDING_SIZE = params['EMBEDDING_SIZE']
    DROPOUT_GNN = params['DROPOUT_GNN']
    DROPOUT_CLASSIFIER = params['DROPOUT_CLASSIFIER']
    RANDOM_SEED = params['RANDOM_SEED'] # Agora controlado pelo loop 'main'
    TIPO_LESÃO = params['TIPO_LESÃO']
    TIPO_SEGMENTAÇÃO = params['TIPO_SEGMENTAÇÃO']
    MAX_DIM_A_TESTAR = params['MAX_DIM_A_TESTAR']
    N_NODES_A_TESTAR = params['N_NODES_A_TESTAR']
    NUM_CLASSES = 2
    TEST_SPLIT_SIZE = 0.10
    VAL_SPLIT_SIZE = 0.10
    EARLY_STOPPING_PATIENCE = 50 

    dim_folder_name = str(MAX_DIM_A_TESTAR)
    
    path_map = {
        ("MASS", "SLIC"): PASTA_FEATURES_MASS_SLIC,
        ("MASS", "DISF"): PASTA_FEATURES_MASS_DISF,
        ("CALC", "SLIC"): PASTA_FEATURES_CALC_SLIC,
        ("CALC", "DISF"): PASTA_FEATURES_CALC_DISF,
    }
    
    try:
        pasta_features = path_map[(TIPO_LESÃO, TIPO_SEGMENTAÇÃO)]
    except KeyError:
        print(f"AVISO: Combinação {TIPO_LESÃO}/{TIPO_SEGMENTAÇÃO} não mapeada. Pulando.")
        return None

    try:
        all_data = resgata_dados_treinamento(pasta_features, N_NODES_A_TESTAR, dim_folder_name)
        all_data = normalizar_features(all_data) 
    except FileNotFoundError:
        return None
    
    # Define o dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)

    # 2. Divisão e Pesos dos Dados
    try:
        stratify_labels = [g.source for g in all_data]
    except AttributeError:
        stratify_labels = [g.y.item() for g in all_data]

    train_val_data, test_data = train_test_split(
        all_data, test_size=TEST_SPLIT_SIZE, stratify=stratify_labels, random_state=RANDOM_SEED
    )
    train_val_stratify_labels = [g.source for g in train_val_data] if 'source' in train_val_data[0] else [g.y.item() for g in train_val_data]
    val_split_adjusted = VAL_SPLIT_SIZE / (1.0 - TEST_SPLIT_SIZE)
    train_data, val_data = train_test_split(
        train_val_data, test_size=val_split_adjusted, stratify=train_val_stratify_labels, random_state=RANDOM_SEED
    )

    # Cálculo dos pesos da classe 
    train_labels_counts = Counter([g.y.item() for g in train_data])
    total_samples = sum(train_labels_counts.values())
    weight_0 = total_samples / (NUM_CLASSES * train_labels_counts.get(0, 1))
    weight_1 = total_samples / (NUM_CLASSES * train_labels_counts.get(1, 1))
    class_weights = torch.tensor([weight_0, weight_1], dtype=torch.float32).to(device)
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. Configurar Modelo e Treinamento
    model = BRM(
        node_feature_size=train_data[0].num_node_features,
        edge_feature_size=train_data[0].num_edge_features,
        num_classes=NUM_CLASSES,
        embedding_size=EMBEDDING_SIZE,
        dropout_gnn=DROPOUT_GNN,
        dropout_classifier=DROPOUT_CLASSIFIER
    ).to(device)


    run_name = f"{TIPO_LESÃO}_{TIPO_SEGMENTAÇÃO}_E{EMBEDDING_SIZE}_LR{LEARNING_RATE}_WD{WEIGHT_DECAY}_DC{DROPOUT_CLASSIFIER}_S{RANDOM_SEED}"
    best_val_accu = 0
    patience_counter = 0
    save_dir = salva_modelo_treinado(dim_folder_name, TIPO_LESÃO, N_NODES_A_TESTAR, run_name)
    
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=15, min_lr=1e-6, factor=0.5)



    for epoch in range(EPOCHS):
        # Passando 'position=1' para o tqdm de época não interferir no tqdm principal
        train_loss = train(epoch=epoch, model=model, train_loader=train_loader, optimizer=optimizer, loss_fn=loss_fn, device=device)
        val_loss, val_accu = val(epoch=epoch, model=model, val_loader=val_loader, loss_fn=loss_fn, device=device, run_type="val")

        if val_accu > best_val_accu:
            best_val_accu = val_accu
            torch.save(model.state_dict(), save_dir)
            patience_counter = 0 
        else:
            patience_counter += 1
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            break 
        
        scheduler.step(val_accu)

    # 4. Avaliação Final
    final_metrics = {'accuracy': 0.0, '1': {'recall': 0.0, 'precision': 0.0, 'f1-score': 0.0, 'support': 0}}
    
    if best_val_accu > 0:
        best_model = BRM(
            node_feature_size=test_data[0].num_node_features,
            edge_feature_size=test_data[0].num_edge_features,
            num_classes=NUM_CLASSES,
            embedding_size=EMBEDDING_SIZE
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

        final_metrics = classification_report(test_labels, test_preds, output_dict=True, zero_division=0)
   
            
        # 5. Resultado para o CSV
        results = {
            # Hiperparâmetros
            'TIPO_LESÃO': TIPO_LESÃO,
            'TIPO_SEGMENTAÇÃO': TIPO_SEGMENTAÇÃO,
            'N_NODES': N_NODES_A_TESTAR,
            'EMBEDDING_SIZE': EMBEDDING_SIZE,
            'LEARNING_RATE': LEARNING_RATE,
            'WEIGHT_DECAY': WEIGHT_DECAY,
            'DROPOUT_GNN': DROPOUT_GNN,
            'DROPOUT_CLASSIFIER': DROPOUT_CLASSIFIER,
            'RANDOM_SEED': RANDOM_SEED, 
            # Métricas
            'CLASS_WEIGHT_MALIGN': class_weights[1].item(),
            'TEST_ACCURACY': final_metrics.get('accuracy', 0.0),

            'TEST_RECALL_MALIGN': final_metrics.get('1', {}).get('recall', 0.0),
            'TEST_PRECISION_MALIGN': final_metrics.get('1', {}).get('precision', 0.0),
            'TEST_F1_MALIGN': final_metrics.get('1', {}).get('f1-score', 0.0),
            'TEST_SUPPORT_MALIGN': final_metrics.get('1', {}).get('support', 0),

            'TEST_RECALL_BENIGN': final_metrics.get('0', {}).get('recall', 0.0),
            'TEST_PRECISION_BENIGN': final_metrics.get('0', {}).get('precision', 0.0),

            'BEST_VAL_ACCURACY': best_val_accu
        }
    else:
        print(f"AVISO: Experimento {run_name} não produziu acurácia de validação > 0. Pulando.")
        return None
        
    return results


def main():
    
    # --- Configuração das Rodadas ---
    SEEDS_TO_RUN = [42, 123, 1024] # 3 rodadas com seeds diferentes
    N_REPEATS = len(SEEDS_TO_RUN)

    keys, values = zip(*param_grid.items())
    experiment_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    total_combinations = len(experiment_combinations)
    print(f"Total de {total_combinations} combinações de hiperparâmetros a testar.")
    print(f"Cada combinação será executada {N_REPEATS} vezes (Seeds: {SEEDS_TO_RUN}).")
    print(f"O melhor resultado (por acurácia) de cada {N_REPEATS} será salvo.")

    all_best_run_results = [] 
    
    diretorio_base = os.path.dirname(os.path.abspath(sys.argv[0])) if hasattr(sys, 'argv') and os.path.exists(os.path.dirname(os.path.abspath(sys.argv[0]))) else os.getcwd()
    csv_path = os.path.join(diretorio_base, "hpo_results_best_of_3.csv") 
    header_written = False

    # Loop principal (tqdm) itera sobre as COMBINAÇÕES de hiperparâmetros
    for combination in tqdm(experiment_combinations, 
                            total=total_combinations,
                            desc=f"Executando {total_combinations} Combinações (x{N_REPEATS} runs)",
                            position=0, dynamic_ncols=True): 
        
        # --- Anuncia a combinação ---
        combo_name = (f"{combination['TIPO_LESÃO']}-{combination['TIPO_SEGMENTAÇÃO']} | "
                      f"LR: {combination['LEARNING_RATE']} | "
                      f"WD: {combination['WEIGHT_DECAY']} | "
                      f"DC: {combination['DROPOUT_CLASSIFIER']}")
        tqdm.write(f"\n--- Testando Combinação: {combo_name} ---")

        per_combination_results = [] # Armazena os resultados das 3 rodadas
        
        # Loop interno: Roda N vezes com seeds diferentes
        for seed in SEEDS_TO_RUN:
            current_params = combination.copy()
            current_params['RANDOM_SEED'] = seed
            
            result = run_experiment_sequential(current_params)
            
            if result is not None:
                # --- Print dos resultados da rodada ---
                tqdm.write(f"    [Run Seed {seed:4d}] "
                      f"Acc: {result['TEST_ACCURACY']:.4f} | "
                      f"Recall (M): {result['TEST_RECALL_MALIGN']:.4f} | "
                      f"Prec (M): {result['TEST_PRECISION_MALIGN']:.4f} | "
                      f"Recall (B): {result['TEST_RECALL_BENIGN']:.4f} | "
                      f"Prec (B): {result['TEST_PRECISION_BENIGN']:.4f}")
                per_combination_results.append(result)

        # --- Seleção do Melhor Resultado ---
        if not per_combination_results:
            tqdm.write(f"AVISO: Nenhuma das {N_REPEATS} rodadas teve sucesso para esta combinação. Pulando.")
            continue

        best_run_dict = max(per_combination_results, key=lambda run: run['TEST_ACCURACY'])
        
        # --- Anuncia o vencedor ---
        tqdm.write(f"    -> Melhor Rodada (Seed {best_run_dict['RANDOM_SEED']}) selecionada com Acurácia: {best_run_dict['TEST_ACCURACY']:.4f}")

        best_run_dict['N_RUNS_SUCCESSFUL'] = len(per_combination_results)
        all_best_run_results.append(best_run_dict)
        
        # --- Salvar no CSV  ---
        df_to_save = pd.DataFrame([best_run_dict])
        if not header_written:
            df_to_save.to_csv(csv_path, index=False, mode='w', header=True)
            header_written = True
        else:
            df_to_save.to_csv(csv_path, index=False, mode='a', header=False)


    # --- Exportar Resultados para CSV ---
    if not all_best_run_results:
        print("\nNenhum experimento rodado com sucesso.")
        sys.exit(0)

    results_df = pd.DataFrame(all_best_run_results)

    print("\n" + "="*80)
    print(f"Busca de Hiperparâmetros Concluída! {len(results_df)} combinações testadas.")
    print(f"Resultados (melhor de {N_REPEATS}) salvos em: {csv_path}")

    # --- Imprime o melhor resultado (focando na MAIOR ACURÁCIA) ---
    if not results_df.empty:
        best_run = results_df.loc[results_df['TEST_ACCURACY'].idxmax()]
        print("\n--- Melhor Combinação Encontrada (Foco: Maior Acurácia de Teste) ---")
        
        # Adicionado métricas benignas e F1
        cols_to_print = [
            'TIPO_LESÃO', 'TIPO_SEGMENTAÇÃO', 'N_NODES', 
            'LEARNING_RATE', 'WEIGHT_DECAY', 'DROPOUT_CLASSIFIER',
            'TEST_ACCURACY',           
            'TEST_RECALL_MALIGN',      
            'TEST_PRECISION_MALIGN',
            'TEST_F1_MALIGN',
            'TEST_RECALL_BENIGN',
            'TEST_PRECISION_BENIGN',
            'RANDOM_SEED'              
        ]
        cols_to_print = [col for col in cols_to_print if col in best_run.index]
        
        print(best_run[cols_to_print])
        print("="*80)