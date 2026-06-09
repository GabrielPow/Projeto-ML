"""Treino do pipeline de familia sonora (papel: IDENTIFICAR).

Este modulo treina o modelo nao-supervisionado que DEFINE as familias
sonoras a partir das features de audio do catalogo:

    StandardScaler  ->  PCA(26 componentes)  ->  KMeans(K=4)

Decisao validada no notebook `cluster_features_audio.ipynb`:
- 26 componentes do PCA = ~95% da variancia das 40 features
- K=4 foi o melhor numero de clusters (Silhouette/Calinski/Davies-Bouldin)
- Validado por um SVM (F1=0.964) que recupera as familias a partir das
  features originais -> os grupos sao reais, nao artefato do PCA.

Roda uma vez (offline) para rotular o catalogo. Para classificar bandas
novas depois, use models/predict_familia_sonora.py (carrega o .joblib
salvo aqui).

Uso:
    python -m ibmecPCDV20522_gabriel_pedro.models.train_familia_sonora
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..features.build_features import FEATURES_AUDIO, get_audio_features
from ..utils import paths

# ── Hiperparametros (conclusao do notebook) ───────────────────────────
N_PCA = 26
N_CLUSTERS = 4
RANDOM_STATE = 42

# Mapa cluster -> nome da familia sonora.
# ATENCAO: o KMeans numera os clusters (0..3) de forma arbitraria. Estes
# nomes nao sao genericos: cada um descreve o PERFIL real do cluster (ver
# a tabela top-5 de `print_cluster_profiles`). Se ao re-treinar a numeracao
# reordenar (ja aconteceu), o perfil do cluster 0 pode deixar de ser
# "quente/encorpado" -> confira a saida ANTES de confiar e ajuste este mapa.
CLUSTER_NAMES = {
    0: "Dinamico & Contrastante",
    1: "Quente & Encorpado",
    2: "Brilhante & Intenso",
    3: "Pulsante & Sombrio",
}

# Nomes curtos das colunas de probabilidade (mesma ordem dos cluster ids 0..3).
PROBA_COLS = {
    0: "proba_Dinamico",
    1: "proba_Quente",
    2: "proba_Brilhante",
    3: "proba_Pulsante",
}


def build_pipeline(n_pca: int = N_PCA,
                   n_clusters: int = N_CLUSTERS,
                   random_state: int = RANDOM_STATE) -> Pipeline:
    """Monta o pipeline (ainda NAO treinado).

    Encadear as 3 etapas num unico Pipeline garante que o scaler e o PCA
    aprendidos no treino sejam aplicados identicamente em qualquer dado
    novo - sem isso, seria facil escalar bandas novas com estatisticas
    erradas e gerar classificacoes inconsistentes.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_pca, random_state=random_state)),
        ("kmeans", KMeans(n_clusters=n_clusters, random_state=random_state,
                          n_init="auto")),
    ])


def print_cluster_profiles(pipeline: Pipeline, X: pd.DataFrame, labels) -> None:
    """Imprime o perfil sonoro de cada cluster para conferir os nomes.

    Mostra, em z-score (dado escalado), as features mais ALTAS e mais
    BAIXAS de cada cluster. E assim que se valida o mapa CLUSTER_NAMES:
    o perfil de cada numero precisa bater com o nome atribuido a ele.
    """
    X_scaled = pipeline["scaler"].transform(X)
    df_scaled = pd.DataFrame(X_scaled, columns=FEATURES_AUDIO)
    df_scaled["cluster"] = labels
    medias = df_scaled.groupby("cluster")[FEATURES_AUDIO].mean()

    print("\n" + "=" * 60)
    print("PERFIL SONORO POR CLUSTER (confira se os nomes batem)")
    print("=" * 60)
    for c in sorted(medias.index):
        perfil = medias.loc[c].sort_values(ascending=False)
        nome = CLUSTER_NAMES.get(c, "(sem nome)")
        print(f"\nCluster {c} -> {nome}")
        print("  ALTAS : ", ", ".join(perfil.head(5).index))
        print("  BAIXAS: ", ", ".join(perfil.tail(5).index))


def fuzzy_proba(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Probabilidade fuzzy de pertencer a cada familia (cluster).

    O K-Means nao tem predict_proba. Derivamos uma afinidade a partir das
    distancias aos centroides via fuzzy c-means membership (m=2):

        u_ik = (1/d_ik^2) / sum_j (1/d_ij^2)

    Quanto mais perto do centroide, maior a probabilidade. O cluster de
    maior probabilidade (argmax) e sempre o centroide mais proximo, ou
    seja, coincide com o rotulo duro do K-Means. Cada linha soma 1.0.

    Retorna array (n_amostras, n_clusters).
    """
    # pipeline[:-1] = scaler + pca (tudo menos o KMeans). transform leva X ao
    # mesmo espaco de 26 componentes em que os centroides foram aprendidos.
    X_pca = pipeline[:-1].transform(X)
    dist = pipeline["kmeans"].transform(X_pca)        # (n, n_clusters)
    inv_sq = 1.0 / np.square(dist + 1e-12)            # +eps evita div/0
    return inv_sq / inv_sq.sum(axis=1, keepdims=True)  # cada linha soma 1.0


def train(df: pd.DataFrame | None = None, save: bool = True) -> tuple[Pipeline, pd.DataFrame]:
    """Treina o pipeline, rotula o catalogo e salva os artefatos.

    Parametros
    ----------
    df : DataFrame, opcional
        Catalogo com as features de audio (+ ID_ARTISTA). Se None, carrega
        de paths.DF_RECOMMENDER.
    save : bool
        Se True, salva o pipeline em .joblib e a tabela de familias em CSV.

    Retorna
    -------
    (pipeline_treinado, df_familia)
        df_familia enxuto: ID_ARTISTA + familia_sonora + 4 probabilidades.
    """
    if df is None:
        df = pd.read_csv(paths.DF_RECOMMENDER)

    X = get_audio_features(df)

    pipeline = build_pipeline()
    labels = pipeline.fit_predict(X)

    # Probabilidade fuzzy a partir das distancias aos centroides do K-Means.
    proba = fuzzy_proba(pipeline, X)

    # Tabela final enxuta: so o que sera salvo (ID + rotulo + probas).
    df_familia = pd.DataFrame({"ID_ARTISTA": df["ID_ARTISTA"].values})
    df_familia["familia_sonora"] = pd.Series(labels).map(CLUSTER_NAMES).values
    for cid, colname in PROBA_COLS.items():
        df_familia[colname] = proba[:, cid].round(4)

    # Sanity check: o argmax das probabilidades bate com o rotulo duro?
    coerente = bool((proba.argmax(axis=1) == labels).all())
    print(f"Coerencia (argmax proba == familia_sonora): {coerente}")

    print("\nDistribuicao das familias sonoras:")
    print(df_familia["familia_sonora"].value_counts())
    print(f"Total de bandas: {len(df_familia)}")

    print_cluster_profiles(pipeline, X, labels)

    if save:
        # Empacotamos pipeline + nomes + features + colunas de proba num so
        # artefato, para o predict_familia_sonora.py ter tudo que precisa
        # sem depender deste modulo.
        artefato = {
            "pipeline": pipeline,
            "cluster_names": CLUSTER_NAMES,
            "proba_cols": PROBA_COLS,
            "features": FEATURES_AUDIO,
        }
        joblib.dump(artefato, paths.PIPELINE_FAMILIA_SONORA)
        df_familia.to_csv(paths.DF_FAMILIA_SONORA, index=False)
        print(f"\nPipeline salvo em: {paths.PIPELINE_FAMILIA_SONORA}")
        print(f"Tabela de familias salva em: {paths.DF_FAMILIA_SONORA}")

    return pipeline, df_familia


if __name__ == "__main__":
    train()
