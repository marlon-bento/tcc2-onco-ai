# Passo a passo para rodar o código

## criando o ambiente

para começar o ambiente usado para evitar conflitos foi escolhido a versão 3.11 do python

passo 1: instalar a versão do python
```
sudo add-apt-repository ppa:deadsnakes/ppa
```

```
sudo apt update
sudo apt install python3.11
```

passo 2: iniciar o ambiente virtual utilizando a versão instalada

```
python3.11 -m venv venv
```

passo 3: entrar na venv

```
source venv/bin/activate
```

passo 4: instalar as dependencias do projeto
```
pip install -r requirements.txt
```

passo 5: instalar o disf no projeto

instalando o make
```
sudo apt-get update
sudo apt-get install -y python3.11-dev build-essential cmake
```
compilando a biblioteca disf
```
cd DISF
make clean
make python3
```

## todas as funções para o treinamento estão presentes no run.py
parar rodar por exemplo um experimento de feature do brm, rodar o comando
```
python run.py gen_features
```
o mesmo se aplica para os outros

a estrutura a se seguire é a seguinte:
- As features geradas são salvas em um arquivo .pt (para evitar ter que extrair features todas as vezes)
- Essas features geradas são carregadas para treinamento, brm usa as features normais e o higsi as hierarquicas