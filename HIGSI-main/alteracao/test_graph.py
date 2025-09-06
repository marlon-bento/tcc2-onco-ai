# app_final.py (versão final compatível com higra v0.6.12)

import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import networkx as nx
from skimage.segmentation import mark_boundaries
from skimage.measure import regionprops
from skimage import graph as sk_graph

# Importa o SEU arquivo como uma biblioteca!
import graph_utils

def processar_e_visualizar(input_image, n_nodes, canonized):
    """
    Função principal da interface. Orquestra as chamadas ao graph_utils
    e monta a visualização final.
    """
    image_np = np.array(input_image)

    # --- PASSO 1: Consumir graph_utils para gerar superpixels ---
    print("Chamando graph_utils.gerar_superpixels()...")
    superpixels = graph_utils.gerar_superpixels(image_np, n_nodes)

    # --- PASSO 2: Consumir graph_utils para gerar a árvore ---
    print("Chamando graph_utils.gerar_arvore_hierarquica() com a lógica antiga...")
    tree, altitudes = graph_utils.gerar_arvore_hierarquica(image_np, superpixels)
    if not canonized:
        tree, node_map = graph_utils.hg.tree_2_binary_tree(tree)
        altitudes = altitudes[node_map]

    # --- PASSO 3: Consumir a função original para os resultados numéricos ---
    print("Chamando graph_utils.MG_superpixel_hierarchy()...")
    h, edges, _, _, _, _, _ = graph_utils.MG_superpixel_hierarchy(image_np, n_nodes, canonized)
    
    # --- Montagem da Visualização ---
    print("Montando a visualização...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Visualização com Higra v0.6.12", fontsize=20)

    # Gráficos 1, 2, 3 (permanecem iguais)
    axes[0, 0].imshow(image_np); axes[0, 0].set_title("1. Imagem Original"); axes[0, 0].axis('off')
    marked_image = mark_boundaries(image_np, superpixels)
    axes[0, 1].imshow(marked_image); axes[0, 1].set_title(f"2. Superpixels (n={n_nodes})"); axes[0, 1].axis('off')
    rag_sk = sk_graph.rag_mean_color(image_np, superpixels)
    props = regionprops(superpixels)
    node_positions = {prop.label: (prop.centroid[1], prop.centroid[0]) for prop in props}
    axes[1, 0].imshow(marked_image)
    nx.draw(rag_sk, pos=node_positions, ax=axes[1, 0], node_size=40, node_color='yellow', edge_color='cyan', width=1.5)
    axes[1, 0].set_title("3. RAG sobre Superpixels"); axes[1, 0].axis('off')

    # Gráfico 4: Hierarquia (usando a chamada compatível com a versão antiga)
    # ######################################
    # ## AQUI ESTÁ A CORREÇÃO FINAL ##
    # ######################################
    graph_utils.hg.draw_hierarchy(tree, altitudes, ax=axes[1, 1], leaf_labels=False)
    axes[1, 1].set_title("4. Árvore de Hierarquia (do Higra v0.6.12)")

    fig.canvas.draw()
    plot_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    plot_image = plot_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    resultados_texto = (
        f"--- Resultados Numéricos (de MG_superpixel_hierarchy) ---\n"
        f"Matriz de features dos nós (h): {h.shape}\n"
        f"Matriz de arestas (edges): {edges.shape}\n"
    )
    
    return plot_image, resultados_texto

# Interface do Gradio (permanece a mesma)
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    # ... (o código da interface não precisa mudar)
    gr.Markdown("# Interface para `graph_utils.py`")
    gr.Markdown("Esta interface consome as funções do seu script `graph_utils.py` para gerar e visualizar os grafos.")
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Arraste sua Imagem Aqui")
            n_nodes_slider = gr.Slider(minimum=20, maximum=500, step=10, value=100, label="Número de Superpixels")
            canonized_checkbox = gr.Checkbox(label="Usar Árvore Canonizada", value=True)
            submit_button = gr.Button("Processar Imagem", variant="primary")
        with gr.Column(scale=2):
            output_image = gr.Image(label="Resultado da Análise Visual")
            output_text = gr.Textbox(label="Resultados Numéricos")
    submit_button.click(
        fn=processar_e_visualizar, 
        inputs=[input_image, n_nodes_slider, canonized_checkbox], 
        outputs=[output_image, output_text]
    )

if __name__ == "__main__":
    demo.launch()