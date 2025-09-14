import os
import torch
import cv2
import pandas as pd
from tqdm import tqdm
from torch_geometric.data import Data
from decouple import config
from prof.grap_utils_new import MG_superpixel_hierarchy
from multiprocessing import Pool, cpu_count

# ======================================================================
ARQUIVO_MANIFEST = config("MANIFEST_FILE_PATH")
PASTA_IMAGENS = config("DATASET_IMAGES_PATH")
PASTA_FEATURES = config("PASTA_FEATURES")

LISTA_MAX_DIM = [256]
LISTA_N_NODES = [10]

def pre_processar_imagem(path, max_dim):
    img = cv2.imread(path)
    if img is None:
        return None
    if max_dim is not None:
        h, w = img.shape[:2]
        if h > max_dim or w > max_dim:
            scale = max_dim / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 🔑 Função que processa UMA imagem (precisa estar no escopo global para multiprocessing funcionar no Windows)
def processar_linha(args):
    row, max_dim, n_nodes = args
    caminho_relativo = row["image file path"]
    partes_caminho = caminho_relativo.split("/")
    caminho_final_encontrado = None

    if len(partes_caminho) > 1:
        nome_pasta = partes_caminho[-2]  # UID
        pasta_do_exame = os.path.join(PASTA_IMAGENS, nome_pasta)
        if os.path.isdir(pasta_do_exame):
            arquivos = [f for f in os.listdir(pasta_do_exame) if f.lower().endswith((".jpeg", ".jpg", ".png"))]
            if arquivos:
                caminho_final_encontrado = os.path.join(pasta_do_exame, arquivos[0])

    if caminho_final_encontrado:
        img = pre_processar_imagem(caminho_final_encontrado, max_dim)
        if img is None:
            return None

        h_feat, edges, edge_feat, _, graph_index, reduced_index, ng = MG_superpixel_hierarchy(
            img, n_nodes=n_nodes, canonized=True
        )
        return Data(
            x=torch.tensor(h_feat, dtype=torch.float),
            edge_index=torch.tensor(edges, dtype=torch.long),
            edge_attr=torch.tensor(edge_feat, dtype=torch.float),
            y=torch.tensor([row["label"]], dtype=torch.long),
            graph_index=torch.tensor(graph_index, dtype=torch.long),
            reduced_index=torch.tensor(reduced_index, dtype=torch.long),
            ng=torch.tensor(ng, dtype=torch.long),
            source=row["pathology"],
            image_file_path=row["image file path"]
        )
    return None


def gerar_features_higsi():
    df = pd.read_csv(ARQUIVO_MANIFEST)
    print(f"Manifesto carregado com {len(df)} imagens.")

    for max_dim in LISTA_MAX_DIM:
        for n_nodes in LISTA_N_NODES:
            dim_folder_name = "original" if max_dim is None else str(max_dim)
            output_folder = os.path.join(PASTA_FEATURES, "HIGSI", dim_folder_name)
            output_file_path = os.path.join(output_folder, f"{n_nodes}.pt")

            if os.path.exists(output_file_path):
                print(f"Pulando experimento já existente: {output_file_path}")
                continue

            os.makedirs(output_folder, exist_ok=True)

            # ⚡ Multiprocessamento
            args = [(row, max_dim, n_nodes) for _, row in df.iterrows()]
            with Pool(processes=4) as pool:
                resultados = list(tqdm(pool.imap_unordered(processar_linha, args), total=len(args)))

            lista_grafos = [r for r in resultados if r is not None]

            if lista_grafos:
                torch.save(lista_grafos, output_file_path, _use_new_zipfile_serialization=False)
                print(f"✅ {len(lista_grafos)} grafos salvos em {output_file_path}")
            else:
                print(f"⚠️ Nenhum grafo gerado para {dim_folder_name}/{n_nodes}")


if __name__ == "__main__":
    gerar_features_higsi()
