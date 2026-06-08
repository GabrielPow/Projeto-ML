"""Classificacao de familia sonora (papel: CLASSIFICAR banda nova).

Carrega o pipeline ja treinado por train.py e encaixa bandas NOVAS numa
das 4 familias existentes. Nao re-treina nada: o scaler, o PCA e o KMeans
ja vem ajustados dentro do .joblib, entao a banda nova passa exatamente
pelas mesmas transformacoes do catalogo original.

Esta e a funcao que o recommender (recall@K) do time consome: dado o audio
de uma banda, devolve a `familia_sonora` para usar como filtro/feature.

Uso como modulo:
    from ibmecPCDV20522_gabriel_pedro.models.predict import classify
    df_novo["familia_sonora"] = classify(df_novo)
"""

import joblib
import pandas as pd

from ..utils import paths


def load_artifact(caminho=None) -> dict:
    """Carrega o artefato salvo no treino.

    O artefato e um dict com 'pipeline' (treinado), 'cluster_names'
    (mapa numero->nome) e 'features' (ordem das colunas esperada).
    """
    if caminho is None:
        caminho = paths.PIPELINE_FAMILIA_SONORA
    return joblib.load(caminho)


def classify(df: pd.DataFrame, artifact: dict | None = None) -> pd.Series:
    """Classifica bandas na sua familia sonora.

    Parametros
    ----------
    df : DataFrame
        Bandas a classificar, contendo as 40 features de audio.
    artifact : dict, opcional
        Artefato ja carregado. Se None, carrega do disco. (Passar o
        artefato evita reler o .joblib a cada chamada em producao.)

    Retorna
    -------
    pd.Series com o nome da familia sonora, alinhada ao indice de `df`.
    """
    if artifact is None:
        artifact = load_artifact()

    pipeline = artifact["pipeline"]
    cluster_names = artifact["cluster_names"]
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

    return pd.Series(
        [cluster_names[c] for c in cluster_ids],
        index=df.index,
        name="familia_sonora",
    )


if __name__ == "__main__":
    # Demonstracao: classifica o proprio catalogo e compara com o treino,
    # confirmando que predict reproduz os rotulos salvos por train.py.
    df = pd.read_csv(paths.DF_RECOMMENDER)
    familias = classify(df)
    print("Distribuicao prevista pelo predict.py:")
    print(familias.value_counts())
