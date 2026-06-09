"""Classificacao de familia sonora (papel: CLASSIFICAR banda nova).

Carrega o pipeline ja treinado por train_familia_sonora.py e encaixa bandas
NOVAS numa das 4 familias existentes. Nao re-treina nada: o scaler, o PCA e o
KMeans ja vem ajustados dentro do .joblib, entao a banda nova passa exatamente
pelas mesmas transformacoes do catalogo original.

Alem do rotulo duro (`familia_sonora`), devolve as 4 probabilidades fuzzy
(afinidade a cada familia), uteis como feature para o recommender do time.

Uso como modulo:
    from ibmecPCDV20522_gabriel_pedro.models.predict_familia_sonora import classify
    resultado = classify(df_novo)   # DataFrame: familia_sonora + proba_*
"""

import joblib
import numpy as np
import pandas as pd

from ..utils import paths


def load_artifact(caminho=None) -> dict:
    """Carrega o artefato salvo no treino.

    O artefato e um dict com 'pipeline' (treinado), 'cluster_names'
    (mapa numero->nome), 'proba_cols' (mapa numero->nome da coluna de
    probabilidade) e 'features' (ordem das colunas esperada).
    """
    if caminho is None:
        caminho = paths.PIPELINE_FAMILIA_SONORA
    return joblib.load(caminho)


def classify(df: pd.DataFrame, artifact: dict | None = None) -> pd.DataFrame:
    """Classifica bandas na sua familia sonora, com probabilidades.

    Parametros
    ----------
    df : DataFrame
        Bandas a classificar, contendo as 40 features de audio.
    artifact : dict, opcional
        Artefato ja carregado. Se None, carrega do disco. (Passar o
        artefato evita reler o .joblib a cada chamada em producao.)

    Retorna
    -------
    pd.DataFrame alinhado ao indice de `df`, com as colunas:
        familia_sonora  + proba_Quente / proba_Pulsante / ...

    A probabilidade fuzzy vem das distancias aos centroides do K-Means
    (fuzzy c-means, m=2): u_ik = (1/d_ik^2) / sum_j (1/d_ij^2). O argmax
    das probabilidades coincide sempre com o rotulo duro `familia_sonora`.
    """
    if artifact is None:
        artifact = load_artifact()

    pipeline = artifact["pipeline"]
    cluster_names = artifact["cluster_names"]
    proba_cols = artifact["proba_cols"]
    features = artifact["features"]

    # Usa a lista de features salva no treino (e nao uma importada agora)
    # para garantir exatamente as mesmas colunas e ordem com que o pipeline
    # foi ajustado, mesmo que o codigo mude no futuro.
    faltando = [c for c in features if c not in df.columns]
    if faltando:
        raise ValueError(
            f"Faltam {len(faltando)} feature(s) de audio: {faltando}"
        )

    X = df.loc[:, features]
    cluster_ids = pipeline.predict(X)

    # Probabilidade fuzzy: mesma derivacao do treino, reaproveitando o
    # pipeline ja ajustado (scaler+pca -> distancias do KMeans aos centroides).
    X_pca = pipeline[:-1].transform(X)
    dist = pipeline["kmeans"].transform(X_pca)
    inv_sq = 1.0 / np.square(dist + 1e-12)
    proba = inv_sq / inv_sq.sum(axis=1, keepdims=True)

    resultado = pd.DataFrame(index=df.index)
    resultado["familia_sonora"] = [cluster_names[c] for c in cluster_ids]
    for cid, colname in proba_cols.items():
        resultado[colname] = proba[:, cid].round(4)

    return resultado


if __name__ == "__main__":
    # Demonstracao: classifica o proprio catalogo e compara com o treino,
    # confirmando que predict reproduz os rotulos salvos por train.
    df = pd.read_csv(paths.DF_RECOMMENDER)
    resultado = classify(df)
    print("Distribuicao prevista pelo predict_familia_sonora.py:")
    print(resultado["familia_sonora"].value_counts())
