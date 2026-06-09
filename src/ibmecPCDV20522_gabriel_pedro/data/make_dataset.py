"""Monta a tabela final do recommender (papel: PREPARAR dados).

Junta numa unica tabela, por ID_ARTISTA, as tres fontes produzidas pelas
etapas anteriores do time:

    df_recommender.csv            -> metadados + 40 features de audio + Nota/OFERTA
    df_recommender_a_with_lda.csv -> 7 componentes LDA de estilo (modelo do colega)
    df_familia_sonora.csv         -> familia_sonora + 4 probabilidades (modelo proprio)

O resultado e salvo em data/processed/df_final.csv e e consumido diretamente
por models/predict.py, que assim nao precisa refazer os merges a cada execucao.

Uso:
    python -m ibmecPCDV20522_gabriel_pedro.data.make_dataset
"""

from __future__ import annotations

import pandas as pd

from ..utils import paths


# Chave comum as tres tabelas. Inteiro e identico nos tres arquivos.
ID_COLUMN = "ID_ARTISTA"


def build_final_dataset() -> pd.DataFrame:
    """Le as tres tabelas, faz inner join por ID_ARTISTA e devolve a final.

    Inner join: so entram artistas presentes nas tres fontes. Como as tres
    tem o mesmo conjunto de IDs (~3890 linhas), na pratica nada e descartado;
    o inner apenas garante que toda linha final tem LDA e familia sonora.
    """
    fontes = {
        "df_recommender": paths.DF_RECOMMENDER,
        "df_recommender_a_with_lda": paths.DF_RECOMMENDER_LDA,
        "df_familia_sonora": paths.DF_FAMILIA_SONORA,
    }

    for nome, caminho in fontes.items():
        if not caminho.exists():
            raise FileNotFoundError(
                f"Fonte '{nome}' nao encontrada: {caminho}"
            )

    df_recommender = pd.read_csv(fontes["df_recommender"])
    df_lda = pd.read_csv(fontes["df_recommender_a_with_lda"])
    df_familia = pd.read_csv(fontes["df_familia_sonora"])

    df_final = (
        df_recommender
        .merge(df_lda, on=ID_COLUMN, how="inner")
        .merge(df_familia, on=ID_COLUMN, how="inner")
        .reset_index(drop=True)
    )

    return df_final


def main() -> None:
    df_final = build_final_dataset()
    paths.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(paths.DF_FINAL, index=False)
    print(
        f"df_final salvo em {paths.DF_FINAL} "
        f"({df_final.shape[0]} linhas, {df_final.shape[1]} colunas)."
    )


if __name__ == "__main__":
    main()
