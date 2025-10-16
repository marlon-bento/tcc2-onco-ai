# run.py
import typer

from features.gerar_features_rag import gerar_features


from treino.treinar_com_brm import main

from treino.treinar_com_gatconv import main as main_gat


# from treino.treinar_random_forest import treinar_rf

from features.gerar_features_hierarquico import gerar_features_hierarquico

from treino.treinar_higsi import main as main_higsi 


app = typer.Typer(help="Ferramenta de linha de comando para o projeto Onco-AI.")


@app.command("gen_features")
def gen_features():
    """
    Executa o script para gerar as features experimentais simples.
    """
    gerar_features()

@app.command("gen_features_hierarquico")
def gen_features_hierarquico():
    """
    Executa o script para gerar as features experimentais hierárquicas.
    """
    gerar_features_hierarquico()

@app.command("train_brm")
def train_brm():
    """
    Inicia o treinamento do modelo principal (BRM) com as features geradas.
    """
    main()
    #main_sem()

@app.command("train_gat")
def train_gat():
    """
    Inicia o treinamento do modelo principal (GATConv) com as features geradas.
    """
    main_gat()

# @app.command("train_rf")
# def train_rf():
#     """
#     Inicia o treinamento do modelo Random Forest.
#     """
#     treinar_rf()

@app.command("train_higsi")
def train_higsi():
    """
    Inicia o treinamento do modelo HIGSI com as features geradas.
    """
    main_higsi()
    
if __name__ == "__main__":
    app()