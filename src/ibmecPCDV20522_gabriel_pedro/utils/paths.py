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
DATA_INTERIM = DATA / "interim"
DATA_PROCESSED = DATA / "processed"

# Arquivos de dados usados pela familia sonora
DF_RECOMMENDER = DATA_PROCESSED / "df_recommender.csv"
# Saida da familia sonora fica em interim: e dado intermediario (rotulo +
# probabilidades) que ainda sera consumido por etapas seguintes.
DF_FAMILIA_SONORA = DATA_INTERIM / "df_familia_sonora.csv"
# Componentes LDA de estilo (modelo do recommender), tambem intermediario.
DF_RECOMMENDER_LDA = DATA_INTERIM / "df_recommender_a_with_lda.csv"

# Tabela final do recommender: junta df_recommender + LDA + familia sonora
# numa unica tabela por ID_ARTISTA. Gerada por data/make_dataset.py e
# consumida diretamente por models/predict.py (sem merges em runtime).
DF_FINAL = DATA_PROCESSED / "df_final.csv"

# Artefatos de modelo (pipelines treinados em .joblib)
MODELS = ROOT / "models"
PIPELINE_FAMILIA_SONORA = MODELS / "familia_sonora_pipeline.joblib"

# Garante que a pasta de modelos exista ao importar este modulo
MODELS.mkdir(parents=True, exist_ok=True)
