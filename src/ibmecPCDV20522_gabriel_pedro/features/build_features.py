"""Definicao das features de audio usadas pelo modelo de familia sonora.

Manter a lista de features em um unico lugar evita divergencia entre o
treino (train.py) e a predicao (predict.py): os dois importam daqui, entao
e impossivel um usar colunas diferentes do outro.
"""

import pandas as pd

# As 40 features de audio extraidas de cada banda.
# A ORDEM importa: o pipeline treinado espera as colunas nesta sequencia.
FEATURES_AUDIO = [
    "tempo_bpm",
    "rms_mean", "rms_std",
    "zcr_mean", "zcr_std",
    "spectral_centroid_mean", "spectral_centroid_std",
    "spectral_bandwidth_mean", "spectral_bandwidth_std",
    "spectral_rolloff_mean", "spectral_rolloff_std",
    "spectral_flatness_mean", "spectral_flatness_std",
    "mfcc_01_mean", "mfcc_02_mean", "mfcc_03_mean", "mfcc_04_mean",
    "mfcc_05_mean", "mfcc_06_mean", "mfcc_07_mean", "mfcc_08_mean",
    "mfcc_09_mean", "mfcc_10_mean", "mfcc_11_mean", "mfcc_12_mean",
    "mfcc_13_mean",
    "chroma_01_mean", "chroma_02_mean", "chroma_03_mean", "chroma_04_mean",
    "chroma_05_mean", "chroma_06_mean", "chroma_07_mean", "chroma_08_mean",
    "chroma_09_mean", "chroma_10_mean", "chroma_11_mean", "chroma_12_mean",
    "onset_strength_mean", "onset_strength_std",
]


def get_audio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Seleciona e ordena as features de audio de um DataFrame.

    Devolve um DataFrame so com as colunas de FEATURES_AUDIO, na ordem
    esperada pelo pipeline. Mantemos como DataFrame (e nao array) para o
    sklearn preservar os nomes das colunas e validar a ordem na predicao.

    Lanca ValueError se faltar alguma feature, deixando o erro explicito
    em vez de o pipeline falhar com uma mensagem confusa la na frente.
    """
    faltando = [col for col in FEATURES_AUDIO if col not in df.columns]
    if faltando:
        raise ValueError(
            f"Faltam {len(faltando)} feature(s) de audio no DataFrame: {faltando}"
        )

    return df.loc[:, FEATURES_AUDIO].copy()
