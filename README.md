# ML Project

**Repo:** ibmecPCDV20522_gabriel_pedro

**Disciplina:** Projeto de Ciência de Dados V 
**Python:** 3.12

## Equipe
**Membros:** Gabriel Pow; Pedro Pintor, Matheus Okamura  
**E-mails:** gabriel.pow2@gmail.com; pedro20pintor@gmail.com; yamamoto_matheus@hotmail.com


## Estrutura do projeto
- `data/raw` : dados originais (não editar)
- `data/interim` : dados intermediários
- `data/processed` : dataset final para modelagem
- `notebooks/` : exploração/EDA (protótipos)
- `src/` : código reutilizável (pipeline)
- `models/` : artefatos treinados (pkl, joblib, etc.)
- `reports/` : relatório final + figuras/tabelas
- `configs/` : configs de experimentos (yaml/json)

## Como rodar (mínimo)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
## **Arquitetura e Pipelines**

- **Objetivo**: Classificar bandas com base em características de áudio (classificação orientada a áudio) e recomendar artistas similares. O sistema usa características de áudio extraídas (MFCCs, atributos espectrais, tempo, RMS, chroma, etc.) para atribuir cada banda a uma `familia_sonora` e calcular componentes LDA que capturam variação de estilo. Este é um classificador e recomendador baseado em áudio — ele não depende apenas de metadados ou auto-tagging heurístico.

- **Fluxo de dados**: Metadados de áudio brutos e características de áudio extraídas são armazenados em [data/raw](data/raw). Os scripts de pré-processamento e o construtor de dataset geram artefatos intermediários em [data/interim](data/interim) e o dataset final usado na inferência em [data/processed](data/processed).

- **Pipeline A (Estilo / LDA)**: Implementado em [models/groova_model.py](models/groova_model.py) e no módulo de preparação de dados. Etapas:
	- Mapear os rótulos originais de `estilo` em grupos de estilo mais amplos (`map_features`).
	- Ajustar uma projeção LDA (Linear Discriminant Analysis) sobre a tabela de características para produzir recursos `lda_component_...` e persistir em `data/interim` (arquivo: [data/interim/df_recommender_a_with_lda.csv](data/interim/df_recommender_a_with_lda.csv)).
	- Opcionalmente treinar um classificador Gradient Boosting no espaço LDA para predição de `estilo` (usado em análise e relatórios em `groova_model.py`).

- **Pipeline B (Familia Sonora)**: Implementado como um artefato treinado separado salvo em [models/familia_sonora_pipeline.joblib](models/familia_sonora_pipeline.joblib). O script de treinamento e o artefato seguem este padrão:
	- Padronizar características de áudio, aplicar redução de dimensionalidade (PCA) e agrupar com KMeans para definir um pequeno conjunto de famílias sonoras.
	- O artefato contém o pipeline de pré-processamento ajustado e metadados do cluster. Na inferência, o módulo [src/ibmecPCDV20522_gabriel_pedro/models/predict_familia_sonora.py](src/ibmecPCDV20522_gabriel_pedro/models/predict_familia_sonora.py) carrega o artefato e:
		- Prediz um rótulo firme `familia_sonora` para cada nova banda.
		- Calcula escores de pertencimento difuso (`proba_*`) derivados das distâncias aos centróides do cluster (usados como recursos suaves no recomendador).

- **Recomendador / Similaridade**: O recomendador final consome uma tabela mesclada `data/processed/df_final.csv` que contém as características originais, componentes LDA e colunas `proba_*` do pipeline de familia_sonora. O módulo [src/ibmecPCDV20522_gabriel_pedro/models/predict.py](src/ibmecPCDV20522_gabriel_pedro/models/predict.py) constrói um embedding a partir das colunas que começam com `lda_component_` e `proba_`, padroniza essas colunas e usa similaridade cosseno para encontrar os vizinhos mais próximos. A API backend [backend/recommender_api.py](backend/recommender_api.py) expõe endpoints para busca, resumo de catálogo e recomendações.

- **Frontend**: A interface em [frontend/music-recommender](frontend/music-recommender) chama a API backend para buscar e solicitar recomendações. O frontend exibe metadados do artista e escores de similaridade produzidos pelo embedding e pelo recomendador.

## **Como ele classifica (importante)**

- `predict_familia_sonora.py` é o classificador de áudio principal: ele espera um DataFrame com as mesmas colunas de características de áudio usadas no treinamento (o pipeline armazena a lista de características). Ele não re-treina em produção — aplica o pipeline salvo de scaler+PCA+KMeans e retorna tanto um rótulo firme (`familia_sonora`) quanto afinidades difusas (`proba_*`). Isso torna a classificação determinística e reproduzível para novas entradas de áudio.

- O sistema enfatiza sinais derivados de áudio (MFCC, espectral, chroma, tempo, RMS, etc.) em vez de depender de metadados textuais ruidosos; componentes LDA e probabilidades difusas da família são combinados para produzir o embedding de similaridade usado pelo recomendador.

## **Execução rápida (mínimo)**
Execute a preparação de dados e depois inicie o servidor da API:

```bash
# a partir da raiz do projeto
python -m src.ibmecPCDV20522_gabriel_pedro.data.make_dataset
python -m backend.recommender_api
```

Em seguida, abra o frontend (veja [frontend/music-recommender/README.md](frontend/music-recommender/README.md)) ou chame os endpoints da API:

- GET `/api/catalog` — resumo do catálogo
- GET `/api/search?q=<query>` — busca simples
- POST `/api/recommendations` — solicitação de recomendações (veja o esquema de payload em `backend/recommender_api.py`)

## **Onde olhar em seguida (ponteiros de código)**
- Construtor de dados e dataset: [src/ibmecPCDV20522_gabriel_pedro/data/make_dataset.py](src/ibmecPCDV20522_gabriel_pedro/data/make_dataset.py)
- Inferência de familia sonora: [src/ibmecPCDV20522_gabriel_pedro/models/predict_familia_sonora.py](src/ibmecPCDV20522_gabriel_pedro/models/predict_familia_sonora.py)
- Lógica do recomendador: [src/ibmecPCDV20522_gabriel_pedro/models/predict.py](src/ibmecPCDV20522_gabriel_pedro/models/predict.py)
- Servidor da API: [backend/recommender_api.py](backend/recommender_api.py)

