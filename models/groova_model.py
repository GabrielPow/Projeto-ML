import os
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Use relative paths from the notebook location (notebooks/ directory)
RAW_DIR = os.path.join("..", "data", "raw")
INTERIM_DIR = os.path.join("..", "data", "interim")
PROCESSED_DIR = os.path.join("..", "data", "processed")

os.makedirs(INTERIM_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

DATA1_PATH = os.path.join(RAW_DIR, "data1.csv")
DATA2_PATH = os.path.join(RAW_DIR, "data2.csv")
DATA3_PATH = os.path.join(RAW_DIR, "data3.xlsx")
REC_PATH = os.path.join(PROCESSED_DIR, "df_recommender.csv")
PIPELINE_A_PATH = os.path.join(INTERIM_DIR, "df_recommender_a.csv")
PIPELINE_B_PATH = os.path.join(INTERIM_DIR, "df_recommender_b.csv")
FINAL_PATH = os.path.join(PROCESSED_DIR, "df_recommender_final.csv")

print("RAW_DIR:", RAW_DIR)
print("INTERIM_DIR:", INTERIM_DIR)
print("PROCESSED_DIR:", PROCESSED_DIR)

# --- SETUP ---

def load_raw_data():
    df1 = pd.read_csv(DATA1_PATH, sep=None, engine='python', on_bad_lines='skip')
    df2 = pd.read_csv(DATA2_PATH, sep=None, engine='python', on_bad_lines='skip')
    df3 = pd.read_excel(DATA3_PATH)
    return df1, df2, df3


def clean_df3_header(df3_raw):
    new_header = df3_raw.iloc[2]
    df3_cleaned = df3_raw.iloc[3:].copy()
    df3_cleaned.columns = new_header

    # Resetar o índice
    df3_cleaned = df3_cleaned.reset_index(drop=True)

    # Remover colunas totalmente vazias (se houver)
    df3_cleaned = df3_cleaned.dropna(axis=1, how='all')

    return df3_cleaned


def merge_pipeline_data(df1, df3_clean):
    df1['ID_ARTISTA'] = df1['ID_ARTISTA'].astype(str)
    df3_clean['ID_ARTISTA'] = df3_clean['ID_ARTISTA'].astype(str)

    # Realizar o merge (inner join por padrão)
    df_merged = pd.merge(df1, df3_clean, on='ID_ARTISTA', how='inner')
    return df_merged


def build_df_recommender(df_merged):
    columns_to_drop = [
        'Instagram', 'Whatsapp 1', 'Whatsapp 2', 'Telefones', 'E-mails',
        'SLUG Imagem Principal', 'Vídeo 1', 'Vídeo 2', 'Vídeo 3', 'Vídeo 4',
        'Vídeo 5', 'Vídeo 6', 'Vídeo 7', 'Imagem 1', 'Imagem 2', 'Imagem 3',
        'Imagem 4', 'Imagem 5', 'Video 1', 'Video 2', 'Video 3', 'Video 4',
        'Video 5', 'Video 6', 'Video 7', 'Video 8', 'Video 9', 'Video 10',
        'Video 11', 'Video 12', 'Video 13', 'Video 14', 'Video 15',
        'Video 16', 'Video 17', 'Video 18', 'Video 19', 'Video 20',
        'Video 21', 'Video 22', 'Video 23', 'Video 24', 'Video 25',
        'Video 26', 'Video 27', 'Video 28', 'Video 29', 'created_at',
        'updated_at', 'Img 1', 'Img 2', 'Img 3', 'Img 4', 'Img 5', 'Img 6',
        'Img 7', 'Img 8', 'Img 9', 'Img 10', 'Img 11', 'Img 12', 'Img 13',
        'Img 14', 'Img 15', 'FOTO', 'Instagram user', 'whatsapp1',
        'whatsapp2', 'whatsapp1_f', '_whatsapp2_f'
    ]

    df_merged = df_merged.drop(columns=columns_to_drop, errors='ignore')

    columns_for_recommender = [
        'ID_ARTISTA', 'artist_name', 'estilo', 'formacao', 'cidade', 'uf',
        'video_title', 'video_channel', 'video_duration_sec', 'segment_start_sec',
        'segment_duration_sec', 'duration_sec', 'sample_rate', 'tempo_bpm',
        'rms_mean', 'rms_std', 'zcr_mean', 'zcr_std',
        'spectral_centroid_mean', 'spectral_centroid_std',
        'spectral_bandwidth_mean', 'spectral_bandwidth_std',
        'spectral_rolloff_mean', 'spectral_rolloff_std',
        'spectral_flatness_mean', 'spectral_flatness_std',
        'mfcc_01_mean', 'mfcc_02_mean', 'mfcc_03_mean', 'mfcc_04_mean',
        'mfcc_05_mean', 'mfcc_06_mean', 'mfcc_07_mean', 'mfcc_08_mean',
        'mfcc_09_mean', 'mfcc_10_mean', 'mfcc_11_mean', 'mfcc_12_mean',
        'mfcc_13_mean', 'chroma_01_mean', 'chroma_02_mean', 'chroma_03_mean',
        'chroma_04_mean', 'chroma_05_mean', 'chroma_06_mean', 'chroma_07_mean',
        'chroma_08_mean', 'chroma_09_mean', 'chroma_10_mean', 'chroma_11_mean',
        'chroma_12_mean', 'onset_strength_mean', 'onset_strength_std',
        'ESTILO_MUSICAL', 'FORMACAO', 'CIDADE', 'UF', 'OFERTA', 'SLUG', 'Nota'
    ]

    df_recommender = df_merged[columns_for_recommender].copy()
    return df_recommender


def run_pipeline():
    df1, df2, df3 = load_raw_data()
    df3_clean = clean_df3_header(df3)
    df_merged = merge_pipeline_data(df1, df3_clean)
    df_recommender = build_df_recommender(df_merged)
    df_recommender.to_csv(REC_PATH, index=False)
    return df_recommender


df_recommender = run_pipeline()
print('df_recommender Shape:', df_recommender.shape)
df_recommender.head(3)

# --- PIPELINE-A ESTILO ---

def map_features(df_recommender):
    estilo_group_map = {
        'Sertanejo': 'Música Brasileira Popular',
        'Sertanejo Universitário': 'Música Brasileira Popular',
        'Piseiro': 'Música Brasileira Popular',
        'Samba': 'Samba & Ritmos Cariocas',
        'Pagode': 'Música Brasileira Popular',
        'Bossa Nova': 'Samba & Ritmos Cariocas',
        'Chorinho': 'Samba & Ritmos Cariocas',
        'MPB': 'MPB & Pop Nacional',
        'Brasilidades': 'Música Brasileira Popular',
        'Axé': 'Música Brasileira Popular',
        'Forró': 'Música Brasileira Popular',
        'Carnaval': 'Samba & Ritmos Cariocas',
        'Banda Baile': 'Música Brasileira Popular',
        'Rock': 'Rock, Blues & Pop Rock',
        'Pop Rock': 'Rock, Blues & Pop Rock',
        'Indie/Alternativo': 'MPB & Pop Nacional',
        'Pop': 'MPB & Pop Nacional',
        'Músicas Internacionais': 'MPB & Pop Nacional',
        'Black Music': 'Black Music, Soul & RAP',
        'Soul': 'Black Music, Soul & RAP',
        'Funk': 'Eletrônica, Funk & DJ',
        'RAP': 'Black Music, Soul & RAP',
        'Blues': 'Rock, Blues & Pop Rock',
        'Jazz': 'Jazz, MPB Instrumental & Clássica',
        'Country': 'Música Brasileira Popular',
        'Folk': 'Jazz, MPB Instrumental & Clássica',
        'Eletrônica': 'Eletrônica, Funk & DJ',
        'DJ': 'Eletrônica, Funk & DJ',
        'Reggae': 'Cover / Tributo & Outros',
        'Gospel': 'Black Music, Soul & RAP',
        'Música Clássica': 'Jazz, MPB Instrumental & Clássica',
        'Música Latina': 'Jazz, MPB Instrumental & Clássica',
        'Ritmos Variados': 'Cover / Tributo & Outros',
        'Cover/Tributo': 'Cover / Tributo & Outros',
        'Músicas para Crianças': 'Jazz, MPB Instrumental & Clássica'
    }

    # Map original estilos into the merged groups

    df_recommender_a = df_recommender.copy()

    df_recommender_a['estilo'] = df_recommender_a['estilo'].map(estilo_group_map).fillna('Cover / Tributo & Outros')

    return df_recommender_a

def fit_lda_features(df, target_column='estilo'):
    if target_column not in df.columns:
        raise ValueError(f'Coluna alvo não encontrada: {target_column}')

    df = df.copy()
    y = df[target_column].astype(str).fillna('missing').astype('category')
    X = df.drop(columns=[target_column], errors='ignore')
    df_id = df['ID_ARTISTA']
    if 'ID_ARTISTA' in X.columns:
        X = X.drop(columns=['ID_ARTISTA'])

    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ],
        remainder='drop'
    )

    lda_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('lda', LinearDiscriminantAnalysis(solver='svd'))
    ])

    lda_pipeline.fit(X, y)
    X_lda = lda_pipeline.transform(X)
    component_names = [f'lda_component_{i+1}' for i in range(X_lda.shape[1])]
    df_lda = pd.DataFrame(X_lda, columns=component_names, index=df.index)

    df_with_id = pd.concat([df_id,df_lda], axis=1)
    return df_with_id, lda_pipeline

df_recommender_a = map_features(df_recommender)
df_recommender_a, lda_pipeline_a = fit_lda_features(df_recommender_a)
df_recommender_a.to_csv(os.path.join(INTERIM_DIR, 'df_recommender_a_with_lda.csv'), index=False)
print('Pipeline A with LDA feature output shape:', df_recommender_a.shape)

# --- PIPELINE-B GRUPOS SONOROS ---

# --- TREINAR MODELO ---