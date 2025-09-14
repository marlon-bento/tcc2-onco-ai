import os
import torch
import numpy as np
import joblib
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import KFold
from decouple import config

PASTA_FEATURES = config("PASTA_FEATURES")
if not PASTA_FEATURES:
    raise ValueError("ERRO: Variável PASTA_FEATURES não encontrada. Verifique seu arquivo .env.")

# ==============================================================================
# --- PAINEL DE CONTROLE DO EXPERIMENTO ---
# ==============================================================================
MAX_DIM = 512
N_NODES = 100
RANDOM_SEED = 42
TEST_SPLIT_SIZE = 0.20 # <-- % do total que vai para teste

N_ESTIMATORS = 200
MAX_DEPTH = 20
# ==============================================================================

def salvar_modelo_rf(model, scaler, dim, nodes):
    # (Esta função permanece a mesma, está correta)
    diretorio_do_script = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(diretorio_do_script, "weights")
    os.makedirs(save_dir, exist_ok=True)
    model_filename = f"RandomForest_{dim}_{nodes}_seed{RANDOM_SEED}.joblib"
    scaler_filename = f"Scaler_RF_{dim}_{nodes}_seed{RANDOM_SEED}.joblib"
    model_path = os.path.join(save_dir, model_filename)
    scaler_path = os.path.join(save_dir, scaler_filename)
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Modelo Random Forest salvo em: {model_path}")
    print(f"Normalizador (Scaler) salvo em: {scaler_path}")

def treinar_rf():
    features_path = os.path.join(PASTA_FEATURES, str(MAX_DIM), f"{N_NODES}.pt")
    print("="*60)
    print("Iniciando Treinamento do Random Forest")
    print(f"  - Carregando features de: {features_path}")
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"ERRO: Arquivo de grafos '{features_path}' não encontrado.")
    
    lista_grafos = torch.load(features_path, weights_only=False)

    print("Agregando features ricas dos grafos (mean, std, max, min)...")
    X_list = []
    for data in lista_grafos:
        mean_features = data.x.mean(dim=0)
        std_features = data.x.std(dim=0)
        max_features = data.x.max(dim=0).values
        min_features = data.x.min(dim=0).values
        
        # Concatena todas as features em um único vetor
        combined_features = torch.cat([mean_features, std_features, max_features, min_features])
        X_list.append(combined_features.numpy())
    y_list = [data.y.item() for data in lista_grafos]
    
    try:
        stratify_labels = [g.source for g in lista_grafos]
        print(f"Contagem de subgrupos para estratificação: {Counter(stratify_labels)}")
    except AttributeError:
        print("AVISO: Atributo '.source' não encontrado. Usando labels binários (y) para estratificação.")
        stratify_labels = y_list
        
    X = np.array(X_list)
    y = np.array(y_list)

    print("Dividindo dados em treino e teste...")
    
    # --- MUDANÇA PRINCIPAL: Divisão única em treino e teste ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SPLIT_SIZE, 
        stratify=stratify_labels,
        random_state=RANDOM_SEED
    )
    
    print(f"Dataset dividido:")
    print(f"  - Treino: {len(X_train)} amostras")
    # A linha de validação foi removida
    print(f"  - Teste: {len(X_test)} amostras")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Treinando RandomForest com {N_ESTIMATORS} estimadores...")
    rf_model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)
    
    salvar_modelo_rf(rf_model, scaler, MAX_DIM, N_NODES)

    print("\nAvaliando no conjunto de teste...")
    y_pred = rf_model.predict(X_test_scaled)

    print("\n--- Relatório de Classificação Final (Random Forest no Teste) ---")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Acurácia Final no Teste: {accuracy:.4f}\n")
    print(classification_report(y_test, y_pred, target_names=["Benigno (0)", "Maligno (1)"]))
    print("="*60)
