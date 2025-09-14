# run.py
import typer

from features.gerar_features_experimentais_time_new import gerar_features

from features.generate_features_higsi import gerar_features_higsi


from treino.treinar_com_features import main

from treino.treinar_com_features_gatconv import main as main_gat


from treino.treinar_random_forest import treinar_rf

from treino.train_higsi import main as main_higsi


app = typer.Typer(help="Ferramenta de linha de comando para o projeto Onco-AI.")


@app.command("gen_features_higsi")
def gen_features_higsi():
    """
    Executa o script para gerar as features experimentais.
    """
    gerar_features_higsi()



@app.command("gen_features")
def gen_features():
    """
    Executa o script para gerar as features experimentais.
    """
    gerar_features()


@app.command("train_gcn")
def train_gcn():
    """
    Inicia o treinamento do modelo principal (GCN) com as features geradas.
    """
    main()

@app.command("train_higsi")
def train_gcn():
    """
    Inicia o treinamento do modelo principal (GCN) com as features geradas.
    """
    main_higsi()


@app.command("train_gat")
def train_gat():
    """
    Inicia o treinamento do modelo principal (GATConv) com as features geradas.
    """
    main_gat()

@app.command("train_rf")
def train_rf():
    """
    Inicia o treinamento do modelo Random Forest.
    """
    treinar_rf()


if __name__ == "__main__":
    app()