import gradio as gr
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
from grap_utils_new import RAG  # seu arquivo com a função
from skimage.color import label2rgb
from skimage.segmentation import mark_boundaries


from PIL import Image

def process_image_pil(pil_img):
    # Redimensiona mantendo proporção para 256px
    max_dim = 256
    original_size = pil_img.size  # (width, height)
    ratio = min(max_dim / original_size[0], max_dim / original_size[1])
    new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
    pil_img_resized = pil_img.resize(new_size, Image.Resampling.LANCZOS)

    # Converte para numpy RGB
    image_np = np.array(pil_img_resized.convert("RGB"))
    n_nodes = 6
    # Chama sua função RAG
    h, edges, edge_features, pos, super_pixel = RAG(image_np, n_nodes)

    # Cria grafo networkx
    G = nx.Graph()
    n_nodes = h.shape[0]
    for i in range(n_nodes):
        G.add_node(i, pos=(pos[i][0], pos[i][1]))
    
    edge_list = edges.T  # transposta para formato [ (s,t), ... ]
    for s,t in edge_list:
        G.add_edge(s, t)

    # Desenha imagem com grafo em cima
    fig, ax = plt.subplots(figsize=(8,6))
    ax.imshow(image_np)
    pos_dict = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos=pos_dict, node_color='r', edge_color='b', node_size=40, ax=ax)
    ax.axis('off')

    # Salva em buffer e retorna imagem PIL para Gradio mostrar
    import io
    buf = io.BytesIO()
    plt.savefig(buf, format='PNG', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    img_out = Image.open(buf)

    plt.imshow(label2rgb(super_pixel, image_np, kind='avg'))
    plt.title(f"Segmentação SLIC ({n_nodes} superpixels)")
    # salva para exibir no Gradio
    buf = io.BytesIO()
    plt.savefig(buf, format='PNG', bbox_inches='tight')
    buf.seek(0)
    img_out_superpixel = Image.open(buf)

    # Opcional: Exibir a imagem com os superpixels para visualização
    plt.imshow(mark_boundaries(image_np, super_pixel))
    plt.title("Imagem com Superpixels")
    plt.axis('off')
    # salva para exibir no Gradio
    buf = io.BytesIO()
    plt.savefig(buf, format='PNG', bbox_inches='tight')
    buf.seek(0)
    img_out_superpixel_boundaries = Image.open(buf)


    return img_out, img_out_superpixel, img_out_superpixel_boundaries


iface = gr.Interface(
    fn=process_image_pil,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Image(type="pil", label="Imagem com grafo"),
        gr.Image(type="pil", label="Superpixels (label2rgb)"),
        gr.Image(type="pil", label="Superpixels com contornos")
    ],
    title="Visualização RAG com graph_utils",
    description="Faz segmentação, constrói grafo e desenha em cima da imagem original."
)

iface.launch()
