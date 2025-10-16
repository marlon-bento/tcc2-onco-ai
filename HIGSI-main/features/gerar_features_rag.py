import os
import time  
from datetime import datetime, timedelta 
import locale
import torch
import cv2
import pandas as pd
from tqdm import tqdm
from torch_geometric.data import Data
from dotenv import load_dotenv
import os
from prof.graph_utils import RAG, RAG_DISF
from multiprocessing import Pool, cpu_count 

from decouple import config

# ==============================================================================
# --- PAINEL DE CONTROLE DO EXPERIMENTO ---
# ==============================================================================
# Variáveis de Configuração (Aqui você altera o cenário que quer rodar!)
# ----------------------------------------------------------------------
# Escolha o tipo de lesão: "MASS" ou "CALC"
TIPO_LESÃO = "CALC" 
# Escolha o tipo de segmentação/grafo: "SLIC" ou "DISF"
TIPO_SEGMENTAÇÃO = "DISF" 

# Lendo caminhos do arquivo .env 
DATASET_IMAGES_PATH = config("DATASET_IMAGES_PATH")
DATASET_MASS = config("DATASET_MASS")
DATASET_CALC = config("DATASET_CALC")

PASTA_FEATURES_MASS_SLIC = config("PASTA_FEATURES_MASS_SLIC")
PASTA_FEATURES_MASS_DISF = config("PASTA_FEATURES_MASS_DISF")
PASTA_FEATURES_CALC_SLIC = config("PASTA_FEATURES_CALC_SLIC")
PASTA_FEATURES_CALC_DISF = config("PASTA_FEATURES_CALC_DISF")
ARQUIVO_MANIFEST = ''
PASTA_FEATURES = ''
# --- Lógica de Seleção de Caminhos ---
if TIPO_LESÃO.upper() == "MASS":
    ARQUIVO_MANIFEST = DATASET_MASS
    if TIPO_SEGMENTAÇÃO.upper() == "SLIC":
        PASTA_FEATURES = PASTA_FEATURES_MASS_SLIC
    elif TIPO_SEGMENTAÇÃO.upper() == "DISF":
        PASTA_FEATURES = PASTA_FEATURES_MASS_DISF
    else:
        raise ValueError("TIPO_SEGMENTAÇÃO deve ser 'SLIC' ou 'DISF'.")
elif TIPO_LESÃO.upper() == "CALC":
    ARQUIVO_MANIFEST = DATASET_CALC
    if TIPO_SEGMENTAÇÃO.upper() == "SLIC":
        PASTA_FEATURES = PASTA_FEATURES_CALC_SLIC
    elif TIPO_SEGMENTAÇÃO.upper() == "DISF":
        PASTA_FEATURES = PASTA_FEATURES_CALC_DISF
    else:
        raise ValueError("TIPO_SEGMENTAÇÃO deve ser 'SLIC' ou 'DISF'.")
else:
    raise ValueError("TIPO_LESÃO deve ser 'MASS' ou 'CALC'.")

if not ARQUIVO_MANIFEST or not DATASET_IMAGES_PATH or not PASTA_FEATURES:
    raise ValueError("ERRO: Um dos caminhos essenciais não foi definido corretamente.")

NUM_WORKERS = 4

LISTA_MAX_DIM = [1024]
LISTA_N_NODES = [25]


def pre_processar_imagem(caminho_final_encontrado, max_dim):
    img_color = cv2.imread(caminho_final_encontrado)
    if img_color is None: return None
    img_processada = img_color.copy()
    
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

def processar_e_gerar_grafo(args):
    row, max_dim, n_nodes = args
    caminho_relativo = row['image file path']
    caminho_final_encontrado = None
    partes_caminho = caminho_relativo.split('/')
    
    if len(partes_caminho) > 1:
        nome_pasta = partes_caminho[-2]
        pasta_do_exame = os.path.join(DATASET_IMAGES_PATH, nome_pasta)
        if os.path.isdir(pasta_do_exame):
            arquivos = [f for f in os.listdir(pasta_do_exame) if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
            if arquivos:
                caminho_final_encontrado = os.path.join(pasta_do_exame, arquivos[0])

    if caminho_final_encontrado:
        img_processada = pre_processar_imagem(caminho_final_encontrado, max_dim)
        if img_processada is None:
            return None
        
        if TIPO_SEGMENTAÇÃO.upper() == "SLIC":
            # Gera o grafo
            h_feat, edges, edge_feat, _, _ = RAG(img_processada, n_nodes=n_nodes)
        elif TIPO_SEGMENTAÇÃO.upper() == "DISF":
            h_feat, edges, edge_feat, _, _ = RAG_DISF(img_processada, n_nodes=n_nodes)
        else:
            raise ValueError("TIPO_SEGMENTAÇÃO deve ser 'SLIC' ou 'DISF'.")
        # Retorna o grafo gerado
        return Data(
            x=torch.tensor(h_feat, dtype=torch.float), 
            edge_index=torch.tensor(edges, dtype=torch.long),
            edge_attr=torch.tensor(edge_feat, dtype=torch.float), 
            y=torch.tensor([row['label']], dtype=torch.long),
            
            source=row['pathology'], image_file_path=row['image file path']
        )
    return None


def gerar_features():
    try:
        locale.setlocale(locale.LC_TIME, 'pt_BR.UTF-8')
    except locale.Error:
        print("Aviso: O locale 'pt_BR.UTF-8' não está instalado. Usando o padrão do sistema.")

    df_manifest = pd.read_csv(ARQUIVO_MANIFEST)
    print(f"Manifesto '{ARQUIVO_MANIFEST}' carregado com {len(df_manifest)} imagens.")

    print("\nCalculando previsão de tempo...")
    
    # 1. Identifica quais experimentos ainda precisam ser rodados
    experimentos_pendentes = []
    for max_dim in LISTA_MAX_DIM:
        for n_nodes in LISTA_N_NODES:
            dim_folder_name = "original" if max_dim is None else str(max_dim)
            output_file_path = os.path.join(PASTA_FEATURES, dim_folder_name, f'{n_nodes}.pt')
            if not os.path.exists(output_file_path):
                experimentos_pendentes.append({'max_dim': max_dim, 'n_nodes': n_nodes})

    if not experimentos_pendentes:
        print("🎉 Todos os experimentos já foram concluídos!")

    for max_dim in LISTA_MAX_DIM:
        for n_nodes in LISTA_N_NODES:
            dim_folder_name = "original" if max_dim is None else str(max_dim)
            output_folder = os.path.join(PASTA_FEATURES, dim_folder_name)
            output_file_path = os.path.join(output_folder, f'{n_nodes}.pt')

            print("\n" + "="*50)
            print(f"Iniciando experimento: Tamanho Imagem = {dim_folder_name}, Nós = {n_nodes}")
            
            if os.path.exists(output_file_path):
                print(f"Resultado já existe. Pulando.")
                continue

            # --- PREPARAÇÃO PARA O PARALELISMO ---
            # Cria a lista de argumentos para cada worker, que é uma tupla (linha do df, max_dim, n_nodes)
            tarefas = [(row, max_dim, n_nodes) for _, row in df_manifest.iterrows()]

            # Cria o pool de workers
            with Pool(processes=NUM_WORKERS) as pool:
                # Mapeia as tarefas para os workers e usa tqdm para mostrar o progresso
                lista_grafos_atual = list(tqdm(
                    pool.imap(processar_e_gerar_grafo, tarefas),
                    total=len(tarefas),
                    desc=f"Processando {dim_folder_name}/{n_nodes}"
                ))

            lista_grafos_atual = [g for g in lista_grafos_atual if g is not None]

            if lista_grafos_atual:
                os.makedirs(output_folder, exist_ok=True)
                torch.save(lista_grafos_atual, output_file_path, _use_new_zipfile_serialization=False)
                print(f"SUCESSO: {len(lista_grafos_atual)} grafos salvos em '{output_file_path}'")
            else:
                print("AVISO: Nenhum grafo foi gerado para esta configuração.")

    print("\n" + "="*50)
    print("Todos os experimentos foram concluídos!")