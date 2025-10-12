import pandas as pd
import os
from decouple import config

# --- FUNÇÃO PRINCIPAL DE VERIFICAÇÃO ---
def verificar_manifesto():
    """Carrega o arquivo manifesto e exibe um resumo dos dados."""
    
    # Tentativa de carregar o caminho do arquivo do config (decouple)
    try:
        ARQUIVO_MANIFEST = config("MANIFEST_FILE_PATH")
    except Exception:
        print("ERRO: Não foi possível carregar 'MANIFEST_FILE_PATH'. Verifique se o arquivo .env está configurado.")
        return

    if not os.path.exists(ARQUIVO_MANIFEST):
        print(f"ERRO: Arquivo manifesto não encontrado no caminho: {ARQUIVO_MANIFEST}")
        return

    print("=" * 70)
    print(f"🔍 VERIFICAÇÃO DO ARQUIVO MANIFESTO: {os.path.basename(ARQUIVO_MANIFEST)}")
    print("=" * 70)
    
    try:
        # Carrega o CSV
        df_manifest = pd.read_csv(ARQUIVO_MANIFEST)
        
        total_linhas = len(df_manifest)
        print(f"✅ Total de linhas/lesões (ROIs): {total_linhas}")
        print("-" * 70)
        
        # 1. Checa a presença das colunas essenciais
        colunas_esperadas = ['image file path', 'pathology', 'label']
        colunas_faltando = [c for c in colunas_esperadas if c not in df_manifest.columns]
        
        if colunas_faltando:
            print(f"❌ ATENÇÃO: Colunas essenciais faltando: {colunas_faltando}")
        else:
            print("✅ Colunas essenciais ('image file path', 'pathology', 'label') presentes.")

        print("-" * 70)
        
        # 2. Resumo da Coluna 'pathology' (Original)
        if 'pathology' in df_manifest.columns:
            print("Contagem da 'pathology' (Original):")
            print(df_manifest['pathology'].value_counts().to_string())
            print("-" * 70)

        # 3. Resumo da Coluna 'label' (Binária 0/1)
        if 'label' in df_manifest.columns:
            print("Contagem da 'label' (Binária):")
            print(df_manifest['label'].value_counts().to_string())
            
            # Confirma se o total de labels bate com o total de linhas
            if df_manifest['label'].value_counts().sum() == total_linhas:
                print(f"  (Total de labels: {df_manifest['label'].value_counts().sum()})")
            else:
                print("❌ ATENÇÃO: Contagem de labels não bate com o total de linhas.")
            print("-" * 70)

        # 4. Amostra dos dados
        print("Exemplo das 5 primeiras linhas:")
        # Seleciona as colunas mais importantes para a extração
        print(df_manifest[['patient_id', 'image file path', 'pathology', 'label']].head().to_string())
        
        print("=" * 70)

    except Exception as e:
        print(f"❌ ERRO ao carregar ou processar o arquivo manifesto: {e}")

if __name__ == "__main__":
    # Carrega as variáveis de ambiente, se necessário (depende da sua configuração local)
    # load_dotenv() 
    verificar_manifesto()