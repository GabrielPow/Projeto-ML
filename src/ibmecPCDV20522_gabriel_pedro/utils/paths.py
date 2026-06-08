"""Caminhos centralizados do projeto.

Em vez de espalhar caminhos relativos como '../data/...' (que quebram
dependendo de onde o codigo e executado), todos os caminhos sao derivados
da raiz do projeto a partir da localizacao deste arquivo. Assim funciona
igual rodando de um notebook, de um script ou de outra pasta.
"""

from pathlib import Path

# Este arquivo: src/ibmecPCDV20522_gabriel_pedro/utils/paths.py
# parents[0]=utils  [1]=pacote  [2]=src  [3]=raiz do projeto
ROOT = Path(__file__).resolve().parents[3]

# Dados
DATA = ROOT / "data"
DATA_RAW = DATA / "raw"
DATA_PROCESSED = DATA / "processed"

# Arquivos de dados usados pela familia sonora
DF_RECOMMENDER = DATA_PROCESSED / "df_recommender.csv"
DF_FAMILIA_SONORA = DATA_PROCESSED / "df_familia_sonora.csv"

# Artefatos de modelo (pipelines treinados em .joblib)
MODELS = ROOT / "models"
PIPELINE_FAMILIA_SONORA = MODELS / "familia_sonora_pipeline.joblib"

# Garante que a pasta de modelos exista ao importar este modulo
MODELS.mkdir(parents=True, exist_ok=True)
