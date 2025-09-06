import tkinter as tk
from tkinter import filedialog, Scale
import time  # <--- NOVO: Importado para o cronômetro
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx

# Importa a função RAG principal do seu arquivo de utilidades
from grap_utils_new import RAG

# Importações para visualização, igual ao Colab
from skimage.color import label2rgb
from skimage.segmentation import find_boundaries
from scipy.ndimage import binary_dilation

# --- Classe da Aplicação com Interface Gráfica (GUI) ---
class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Processador de Imagens com RAG (HIGSI)")
        self.root.geometry("1200x850")

        control_frame = tk.Frame(root)
        control_frame.pack(pady=10)

        self.btn_select = tk.Button(control_frame, text="Selecionar Imagem", command=self.select_image)
        self.btn_select.pack(side=tk.LEFT, padx=10)

        self.n_nodes_label = tk.Label(control_frame, text="Número de Nós (Superpixels):")
        self.n_nodes_label.pack(side=tk.LEFT, padx=10)
        self.n_nodes_slider = Scale(control_frame, from_=4, to=100, orient=tk.HORIZONTAL, length=200)
        self.n_nodes_slider.set(10)
        self.n_nodes_slider.pack(side=tk.LEFT)

        # <--- NOVO: Label para exibir o tempo de processamento ---
        self.timer_label = tk.Label(control_frame, text="Tempo de processamento: --", font=("Arial", 10))
        self.timer_label.pack(side=tk.LEFT, padx=20)
        # --------------------------------------------------------

        plot_frame = tk.Frame(root)
        plot_frame.pack(fill=tk.BOTH, expand=True)

        self.fig1, self.axs1 = plt.subplots(1, 4, figsize=(16, 4))
        self.fig1.suptitle("Etapas de Pré-Processamento", fontsize=16)
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=plot_frame)
        self.canvas1.get_tk_widget().pack(pady=10)

        self.fig2, self.axs2 = plt.subplots(1, 4, figsize=(16, 4))
        self.fig2.suptitle("Análise RAG - Visualização Estilo Colab", fontsize=16)
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=plot_frame)
        self.canvas2.get_tk_widget().pack(pady=10)

    # <--- NOVO: Função para formatar a duração do tempo ---
    def format_duration(self, seconds):
        """Converte segundos para um formato legível (h, m, s)."""
        if seconds < 60:
            return f"{seconds:.2f}s"
        
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)

        if hours > 0:
            return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        else:
            return f"{int(minutes)}m {int(seconds)}s"
    # ----------------------------------------------------

    def select_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp *.tif")]
        )
        if not file_path:
            return
        
        n_nodes = self.n_nodes_slider.get()
        self.process_and_display_image(file_path, n_nodes)

    def process_and_display_image(self, file_path, n_nodes):
        # <--- NOVO: Inicia o cronômetro ---
        start_time = time.monotonic()
        self.timer_label.config(text="Processando...")
        self.root.update_idletasks() # Força a atualização da interface
        # ----------------------------------

        # --- ETAPA 1: PRÉ-PROCESSAMENTO (IGUAL AO COLAB) ---
        img_color = cv2.imread(file_path)
        if img_color is None:
            self.timer_label.config(text="Erro ao carregar imagem!")
            return

        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_clahe = clahe.apply(img_gray)
        img_median = cv2.medianBlur(img_clahe, ksize=5)

        for ax in self.axs1: ax.clear()
        self.axs1[0].imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)); self.axs1[0].set_title('1. Original')
        self.axs1[1].imshow(img_gray, cmap='gray'); self.axs1[1].set_title('2. Escala de Cinza')
        self.axs1[2].imshow(img_clahe, cmap='gray'); self.axs1[2].set_title('3. CLAHE')
        self.axs1[3].imshow(img_median, cmap='gray'); self.axs1[3].set_title('4. Filtro Mediano')
        for ax in self.axs1: ax.axis('off')
        self.fig1.tight_layout(rect=[0, 0, 1, 0.96])
        self.canvas1.draw()

        # --- ETAPA 2: ANÁLISE RAG (FLUXO DO COLAB) ---
        img_rgb_for_rag = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
        h, edges, edge_features, pos_features, super_pixel = RAG(img_rgb_for_rag, n_nodes=n_nodes)
        
        print(f"Processamento com {n_nodes} nós concluído.")
        print("Shape of node features (h):", h.shape)
        
        for ax in self.axs2: ax.clear()

        self.axs2[0].imshow(img_rgb_for_rag); self.axs2[0].set_title("1. Imagem para RAG")
        self.axs2[1].imshow(label2rgb(super_pixel, img_rgb_for_rag, kind='avg')); self.axs2[1].set_title(f"2. SLIC ({n_nodes} nós)")

        boundaries = find_boundaries(super_pixel)
        for _ in range(4): boundaries = binary_dilation(boundaries)
        img_com_contorno = img_rgb_for_rag.copy()
        img_com_contorno[boundaries] = [255, 0, 0]
        self.axs2[2].imshow(img_com_contorno); self.axs2[2].set_title("3. Contornos")

        G = nx.Graph()
        node_positions = {i: (pos_features[i][0], pos_features[i][1]) for i in range(h.shape[0])}
        G.add_edges_from(edges.T)
        self.axs2[3].imshow(img_rgb_for_rag)
        nx.draw(G, pos=node_positions, node_color='red', edge_color='blue', node_size=50, ax=self.axs2[3])
        self.axs2[3].set_title("4. Grafo (RAG)")

        for ax in self.axs2: ax.axis('off')
        self.fig2.tight_layout(rect=[0, 0, 1, 0.96])
        self.canvas2.draw()

        # <--- NOVO: Para o cronômetro e atualiza o label ---
        end_time = time.monotonic()
        duration = end_time - start_time
        formatted_time = self.format_duration(duration)
        self.timer_label.config(text=f"Tempo de processamento: {formatted_time}")
        # ----------------------------------------------------

# --- Bloco principal para iniciar a aplicação ---
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()