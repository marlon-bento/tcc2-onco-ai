import os
import time
from datetime import datetime, timedelta
import locale
import torch
import cv2
import pandas as pd
from tqdm import tqdm
from torch_geometric.data import Data
from decouple import config
from multiprocessing import Pool, cpu_count # Importa as ferramentas de paralelismo

# Importe a função correta e a classe de dados do seu arquivo de utilidades
# ATENÇÃO: Verifique se o caminho "prof.graph_utils" está correto
from prof.graph_utils import MG_superpixel_hierarchy

# ==============================================================================
# --- DEFINIÇÃO DA CLASSE MultiGraphData ---
# ==============================================================================
class MultiGraphData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'reduced_index':
            return 1
        if key == 'graph_index':
            return self.ng.item()
        return super().__inc__(key, value, *args, **kwargs)

# ==============================================================================
# --- PAINEL DE CONTROLE DO EXPERIMENTO ---
# ==============================================================================
ARQUIVO_MANIFEST = config("MANIFEST_FILE_PATH")
PASTA_IMAGENS = config("DATASET_IMAGES_PATH")
PASTA_FEATURES = config("PASTA_FEATURES_HIGSI")

# --- CONTROLE DE PARALELISMO ---
# O usuário pediu 4, mas você pode ajustar ou usar todos os disponíveis.
NUM_WORKERS = 4  # <<-- AQUI VOCÊ DEFINE QUANTOS NÚCLEOS USAR
# Para usar o máximo de núcleos disponíveis, descomente a linha abaixo:
# NUM_WORKERS = cpu_count()


if not ARQUIVO_MANIFEST or not PASTA_IMAGENS or not PASTA_FEATURES:
    raise ValueError("ERRO: Verifique as variáveis de caminho no arquivo .env.")

LISTA_MAX_DIM = [512]
LISTA_N_NODES = [25,50]

# ==============================================================================
# --- FUNÇÕES AUXILIARES (sem alterações) ---
# ==============================================================================
def formatar_tempo(segundos):
    horas = int(segundos // 3600)
    minutos = int((segundos % 3600) // 60)
    segundos = int(segundos % 60)
    if horas > 0:
        return f"{horas}h, {minutos}m e {segundos}s"
    elif minutos > 0:
        return f"{minutos}m e {segundos}s"
    else:
        return f"{segundos}s"

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
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_channel_clahe = clahe.apply(l_channel)
    lab_clahe = cv2.merge((l_channel_clahe, a_channel, b_channel))
    img_rgb_final = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
    
    return img_rgb_final

# ==============================================================================
# --- NOVA FUNÇÃO DE TRABALHO PARA PROCESSAMENTO PARALELO ---
# ==============================================================================
def processar_linha(args):
    """
    Processa uma única linha do manifest. Esta função será executada em paralelo.
    """
    index, row, max_dim, n_nodes = args
    
    caminho_relativo = row['image file path']
    caminho_final_encontrado = None
    partes_caminho = caminho_relativo.split('/')

    if len(partes_caminho) > 1:
        nome_pasta = partes_caminho[-2]
        pasta_do_exame = os.path.join(PASTA_IMAGENS, nome_pasta)
        if os.path.isdir(pasta_do_exame):
            arquivos = [f for f in os.listdir(pasta_do_exame) if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
            if arquivos:
                caminho_final_encontrado = os.path.join(pasta_do_exame, arquivos[0])

    if caminho_final_encontrado:
        img_processada = pre_processar_imagem(caminho_final_encontrado, max_dim)
        if img_processada is None:
            return None
        
        # O trabalho pesado de gerar o grafo
        h_feat, edges, edge_feat, pos, graph_idx, reduced_idx, n_graphs = MG_superpixel_hierarchy(
            img_processada, n_nodes=n_nodes
        )
        
        # Cria o objeto de dados final
        data = MultiGraphData(
            x=torch.from_numpy(h_feat).type(torch.FloatTensor),
            edge_index=torch.from_numpy(edges).type(torch.LongTensor),
            edge_attr=torch.from_numpy(edge_feat).type(torch.FloatTensor),
            y=torch.tensor([row['label']], dtype=torch.long),
            pos=torch.from_numpy(pos),
            graph_index=torch.from_numpy(graph_idx).type(torch.LongTensor),
            reduced_index=torch.from_numpy(reduced_idx).type(torch.LongTensor),
            ng=torch.from_numpy(n_graphs),
            source=row['pathology'],
            image_file_path=row['image file path']
        )
        return data
        
    return None

# ==============================================================================
# --- BLOCO DE EXECUÇÃO (MODIFICADO PARA PARALELISMO) ---
# ==============================================================================
def gerar_features_higsi():
    try:
        locale.setlocale(locale.LC_TIME, 'pt_BR.UTF-8')
    except locale.Error:
        print("Aviso: O locale 'pt_BR.UTF-8' não está instalado.")
        
    df_manifest = pd.read_csv(ARQUIVO_MANIFEST)
    print(f"Manifesto '{ARQUIVO_MANIFEST}' carregado com {len(df_manifest)} imagens.")
    print(f"Iniciando processamento paralelo com {NUM_WORKERS} workers (núcleos).")

    for max_dim in LISTA_MAX_DIM:
        for n_nodes in LISTA_N_NODES:
            dim_folder_name = "original" if max_dim is None else str(max_dim)
            output_folder = os.path.join(PASTA_FEATURES, dim_folder_name)
            output_file_path = os.path.join(output_folder, f'{n_nodes}.pt')

            print("\n" + "="*50)
            print(f"Iniciando experimento HIGSI: Tamanho Imagem = {dim_folder_name}, Nós = {n_nodes}")
            
            if os.path.exists(output_file_path):
                print(f"Resultado já existe. Pulando.")
                continue

            # Prepara os argumentos para cada chamada da função de trabalho
            tasks = [(index, row, max_dim, n_nodes) for index, row in df_manifest.iterrows()]
            
            lista_grafos_atual = []
            
            # --- AQUI ESTÁ A MÁGICA DO PARALELISMO ---
            # Usamos um 'with' para garantir que os processos do pool sejam encerrados corretamente
            with Pool(processes=NUM_WORKERS) as pool:
                # 'imap_unordered' processa as tarefas em paralelo e retorna os resultados
                # assim que ficam prontos, o que é ideal para a barra de progresso.
                # Envolvemos com tqdm para visualizar o progresso.
                results_iterator = pool.imap_unordered(processar_linha, tasks)
                
                for data in tqdm(results_iterator, total=len(df_manifest), desc=f"Processando {dim_folder_name}/{n_nodes}"):
                    # Adiciona o resultado à lista somente se não for None
                    if data is not None:
                        lista_grafos_atual.append(data)

            if lista_grafos_atual:
                os.makedirs(output_folder, exist_ok=True)
                torch.save(lista_grafos_atual, output_file_path, _use_new_zipfile_serialization=False)
                print(f"SUCESSO: {len(lista_grafos_atual)} grafos hierárquicos salvos em '{output_file_path}'")
            else:
                print("AVISO: Nenhum grafo foi gerado para esta configuração.")

    print("\n" + "="*50)
    print("Todos os experimentos HIGSI foram concluídos!")

