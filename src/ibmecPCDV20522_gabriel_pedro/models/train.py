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
novas depois, use models/predict.py (carrega o .joblib salvo aqui).

Uso:
    python -m ibmecPCDV20522_gabriel_pedro.models.train
"""

import joblib
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
# nomes valem para o treino que gerou os perfis abaixo. Sempre confira a
# saida de `print_cluster_profiles` ao re-treinar: se o perfil do cluster 0
# deixar de ser "quente/encorpado", ajuste este mapa antes de confiar nele.
CLUSTER_NAMES = {
    0: "Dinamico & Contrastante",
    1: "Quente & Encorpado",
    2: "Brilhante & Intenso",
    3: "Pulsante & Sombrio",
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
        print("  ALTO : ", ", ".join(perfil.head(3).index))
        print("  BAIXO: ", ", ".join(perfil.tail(3).index))


def train(df: pd.DataFrame | None = None, save: bool = True) -> tuple[Pipeline, pd.DataFrame]:
    """Treina o pipeline, rotula o catalogo e salva os artefatos.

    Parametros
    ----------
    df : DataFrame, opcional
        Catalogo com as features de audio. Se None, carrega de
        paths.DF_RECOMMENDER.
    save : bool
        Se True, salva o pipeline em .joblib e o catalogo rotulado em CSV.

    Retorna
    -------
    (pipeline_treinado, df_rotulado)
    """
    if df is None:
        df = pd.read_csv(paths.DF_RECOMMENDER)

    X = get_audio_features(df)

    pipeline = build_pipeline()
    labels = pipeline.fit_predict(X)

    # Catalogo rotulado: id numerico do cluster + nome da familia
    df_rotulado = df.copy()
    df_rotulado["cluster_id"] = labels
    df_rotulado["familia_sonora"] = df_rotulado["cluster_id"].map(CLUSTER_NAMES)

    print("Distribuicao das familias sonoras:")
    print(df_rotulado["familia_sonora"].value_counts())

    print_cluster_profiles(pipeline, X, labels)

    if save:
        # Empacotamos pipeline + nomes + features num so artefato, para o
        # predict.py ter tudo que precisa sem depender deste modulo.
        artefato = {
            "pipeline": pipeline,
            "cluster_names": CLUSTER_NAMES,
            "features": FEATURES_AUDIO,
        }
        joblib.dump(artefato, paths.PIPELINE_FAMILIA_SONORA)
        df_rotulado.to_csv(paths.DF_FAMILIA_SONORA, index=False)
        print(f"\nPipeline salvo em: {paths.PIPELINE_FAMILIA_SONORA}")
        print(f"Catalogo rotulado salvo em: {paths.DF_FAMILIA_SONORA}")

    return pipeline, df_rotulado


if __name__ == "__main__":
    train()
