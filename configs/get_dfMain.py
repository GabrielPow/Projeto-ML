import gdown
import pandas as pd

PATH_DATA1 = '../data/raw/data1.csv'
PATH_DATA2 = '../data/raw/data2.csv'
PATH_DATA3 = '../data/raw/data3.csv'

def get_dfMain():

    id1 = '1QB5i4NkAqmLlEmW32K_QCv5MSfTCe6yC'
    id2 = '1sWdxi1BEtBcMwA-3glUBbYZs-lDea0fD'
    id3 = '1ZRNiPFvhqAwsVYBN3oao_fenmRGxCd3-'

    gdown.download(f'https://drive.google.com/uc?id={id1}', PATH_DATA1, quiet=False)
    gdown.download(f'https://drive.google.com/uc?id={id2}', PATH_DATA2, quiet=False)
    gdown.download(f'https://drive.google.com/uc?id={id3}', PATH_DATA3, quiet=False)

    try:
        df1 = pd.read_csv(PATH_DATA1, sep=None, engine='python', on_bad_lines='skip')
    except Exception as e: print(f'Erro no Dataset 1: {e}')

    try:
        df2 = pd.read_csv(PATH_DATA2, sep=None, engine='python', on_bad_lines='skip')
    except Exception as e: print(f'Erro no Dataset 2: {e}')

    try:
        df3 = pd.read_excel(PATH_DATA3)
    except Exception as e: print(f'Erro no Dataset 3: {e}')

    new_header = df3.iloc[2]
    df3_cleaned = df3.iloc[3:].copy()
    df3_cleaned.columns = new_header
    df3_cleaned = df3_cleaned.reset_index(drop=True)
    df3_cleaned = df3_cleaned.dropna(axis=1, how='all')

    df1['ID_ARTISTA'] = df1['ID_ARTISTA'].astype(str)
    df3_cleaned['ID_ARTISTA'] = df3_cleaned['ID_ARTISTA'].astype(str)

    df_merged = pd.merge(df1, df3_cleaned, on='ID_ARTISTA', how='inner')

    columns_to_drop = [
        'Instagram','Whatsapp 1','Whatsapp 2','Telefones','E-mails',
        'SLUG Imagem Principal','Vídeo 1','Vídeo 2','Vídeo 3','Vídeo 4',
        'Vídeo 5','Vídeo 6','Vídeo 7','Imagem 1','Imagem 2','Imagem 3',
        'Imagem 4','Imagem 5','Video 1','Video 2','Video 3','Video 4',
        'Video 5','Video 6','Video 7','Video 8','Video 9','Video 10',
        'Video 11','Video 12','Video 13','Video 14','Video 15','Video 16',
        'Video 17','Video 18','Video 19','Video 20','Video 21','Video 22',
        'Video 23','Video 24','Video 25','Video 26','Video 27','Video 28',
        'Video 29','created_at','updated_at','Img 1','Img 2','Img 3',
        'Img 4','Img 5','Img 6','Img 7','Img 8','Img 9','Img 10','Img 11',
        'Img 12','Img 13','Img 14','Img 15','FOTO','Instagram user',
        'whatsapp1','whatsapp2','whatsapp1_f','_whatsapp2_f'
    ]
    df_merged = df_merged.drop(columns=columns_to_drop, errors='ignore')

    columns_for_recommender = [
        'ID_ARTISTA','artist_name','estilo','formacao','cidade','uf',
        'video_title','video_channel','video_duration_sec','segment_start_sec',
        'segment_duration_sec','duration_sec','sample_rate','tempo_bpm',
        'rms_mean','rms_std','zcr_mean','zcr_std','spectral_centroid_mean',
        'spectral_centroid_std','spectral_bandwidth_mean','spectral_bandwidth_std',
        'spectral_rolloff_mean','spectral_rolloff_std','spectral_flatness_mean',
        'spectral_flatness_std','mfcc_01_mean','mfcc_02_mean','mfcc_03_mean',
        'mfcc_04_mean','mfcc_05_mean','mfcc_06_mean','mfcc_07_mean','mfcc_08_mean',
        'mfcc_09_mean','mfcc_10_mean','mfcc_11_mean','mfcc_12_mean','mfcc_13_mean',
        'chroma_01_mean','chroma_02_mean','chroma_03_mean','chroma_04_mean',
        'chroma_05_mean','chroma_06_mean','chroma_07_mean','chroma_08_mean',
        'chroma_09_mean','chroma_10_mean','chroma_11_mean','chroma_12_mean',
        'onset_strength_mean','onset_strength_std','ESTILO_MUSICAL','FORMACAO',
        'CIDADE','UF','OFERTA','SLUG','Nota'
    ]
    df_recommender = df_merged[columns_for_recommender].copy()
    df_recommender.to_csv('../data/processed/df_recommender.csv')
    return df_recommender

if __name__ == '__main__':
    get_dfMain()