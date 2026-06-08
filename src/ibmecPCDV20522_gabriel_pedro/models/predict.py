from __future__ import annotations

from difflib import get_close_matches
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT_DIR / 'data'
PROCESSED_PATH = DATA_DIR / 'processed' / 'df_recommender.csv'
LDA_PATH = DATA_DIR / 'interim' / 'df_recommender_a_with_lda.csv'

STYLE_GROUP_MAP = {
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
    'Músicas para Crianças': 'Jazz, MPB Instrumental & Clássica',
}


def _normalize_text(value: object) -> str:
    return str(value).strip().casefold()


@lru_cache(maxsize=1)
def load_catalog() -> tuple[pd.DataFrame, list[str], np.ndarray]:
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(f'Arquivo não encontrado: {PROCESSED_PATH}')

    if not LDA_PATH.exists():
        raise FileNotFoundError(f'Arquivo não encontrado: {LDA_PATH}')

    df_recommender = pd.read_csv(PROCESSED_PATH)
    df_lda = pd.read_csv(LDA_PATH)

    catalog = df_recommender.merge(df_lda, on='ID_ARTISTA', how='inner')
    catalog = catalog.reset_index(drop=True)
    catalog['style_group'] = catalog['estilo'].map(STYLE_GROUP_MAP).fillna('Cover / Tributo & Outros')

    lda_columns = [column for column in catalog.columns if column.startswith('lda_component_')]
    if not lda_columns:
        raise ValueError('Os componentes LDA não foram encontrados no catálogo.')

    embeddings = catalog[lda_columns].apply(pd.to_numeric, errors='coerce').fillna(0.0).to_numpy(dtype=float)
    embeddings = StandardScaler().fit_transform(embeddings)

    catalog['artist_name_normalized'] = catalog['artist_name'].map(_normalize_text)
    catalog['video_title_normalized'] = catalog['video_title'].map(_normalize_text)
    catalog['slug_normalized'] = catalog['SLUG'].map(_normalize_text)

    return catalog, lda_columns, embeddings


def search_artists(query: str, limit: int = 6) -> list[dict[str, object]]:
    catalog, _, _ = load_catalog()
    normalized_query = _normalize_text(query)

    if not normalized_query:
        preview = catalog.sort_values(['Nota', 'artist_name'], ascending=[False, True]).head(limit)
        return [_artist_summary(row) for _, row in preview.iterrows()]

    mask = (
        catalog['artist_name_normalized'].str.contains(normalized_query, na=False)
        | catalog['video_title_normalized'].str.contains(normalized_query, na=False)
        | catalog['slug_normalized'].str.contains(normalized_query, na=False)
    )

    matches = catalog.loc[mask].copy()

    if matches.empty:
        close_matches = get_close_matches(normalized_query, catalog['artist_name_normalized'].tolist(), n=limit, cutoff=0.55)
        if not close_matches:
            return []

        matches = catalog[catalog['artist_name_normalized'].isin(close_matches)].copy()

    matches = matches.sort_values(['Nota', 'artist_name'], ascending=[False, True]).head(limit)
    return [_artist_summary(row) for _, row in matches.iterrows()]


def build_catalog_summary(limit: int = 6) -> dict[str, object]:
    catalog, _, _ = load_catalog()
    top_artists = catalog.sort_values(['Nota', 'artist_name'], ascending=[False, True]).head(limit)
    formations = sorted([value for value in catalog['FORMACAO'].dropna().astype(str).unique().tolist() if value])

    return {
        'total_artists': int(catalog.shape[0]),
        'style_groups': int(catalog['style_group'].nunique()),
        'style_groups_options': sorted([value for value in catalog['style_group'].dropna().astype(str).unique().tolist() if value]),
        'formations': formations,
        'top_artists': [_artist_summary(row) for _, row in top_artists.iterrows()],
    }


def recommend_by_artist(
    query: str,
    top_k: int = 6,
    style_group: str | None = None,
    formation: str | None = None,
    bpm_min: float | None = None,
    bpm_max: float | None = None,
    min_note: float | None = None,
) -> dict[str, object]:
    catalog, _, embeddings = load_catalog()
    normalized_query = _normalize_text(query)

    exact_match = catalog[
        (catalog['artist_name_normalized'] == normalized_query)
        | (catalog['video_title_normalized'] == normalized_query)
        | (catalog['slug_normalized'] == normalized_query)
    ]

    if exact_match.empty:
        exact_match = catalog[
            catalog['artist_name_normalized'].str.contains(normalized_query, na=False)
            | catalog['video_title_normalized'].str.contains(normalized_query, na=False)
            | catalog['slug_normalized'].str.contains(normalized_query, na=False)
        ]

    if exact_match.empty:
        return {
            'found': False,
            'query': query,
            'suggestions': search_artists(query, limit=top_k),
            'recommendations': [],
        }

    matched_row = exact_match.iloc[0]
    matched_index = int(matched_row.name)
    similarity_vector = _score_candidates(
        catalog=catalog,
        embeddings=embeddings,
        matched_index=matched_index,
        query_artist_name=matched_row['artist_name'],
        style_group=style_group,
        formation=formation,
        bpm_min=bpm_min,
        bpm_max=bpm_max,
        min_note=min_note,
        top_k=top_k,
    )

    recommendation_rows = similarity_vector['rows']
    predicted_group = similarity_vector['predicted_group']
    matched_score = similarity_vector['matched_score']

    return {
        'found': True,
        'query': query,
        'matched_score': matched_score,
        'predicted_style_group': predicted_group,
        'matched_artist': _artist_detail(matched_row),
        'recommendations': [
            _recommendation_detail(row, score)
            for row, score in recommendation_rows
        ],
    }


def _score_candidates(
    catalog: pd.DataFrame,
    embeddings: np.ndarray,
    matched_index: int,
    query_artist_name: str,
    style_group: str | None,
    formation: str | None,
    bpm_min: float | None,
    bpm_max: float | None,
    min_note: float | None,
    top_k: int,
) -> dict[str, object]:
    matched_vector = embeddings[matched_index]
    distances = np.linalg.norm(embeddings - matched_vector, axis=1)

    candidate_mask = np.ones(len(catalog), dtype=bool)
    candidate_mask[matched_index] = False

    query_artist_norm = _normalize_text(query_artist_name)
    candidate_mask &= catalog['artist_name_normalized'] != query_artist_norm

    if style_group:
        candidate_mask &= catalog['style_group'].astype(str) == style_group

    if formation:
        candidate_mask &= catalog['FORMACAO'].astype(str) == formation

    tempo_bpm = pd.to_numeric(catalog['tempo_bpm'], errors='coerce')

    if bpm_min is not None:
        candidate_mask &= tempo_bpm >= float(bpm_min)

    if bpm_max is not None:
        candidate_mask &= tempo_bpm <= float(bpm_max)

    if min_note is not None:
        candidate_mask &= pd.to_numeric(catalog['Nota'], errors='coerce').fillna(-np.inf) >= float(min_note)

    filtered_catalog = catalog.loc[candidate_mask].copy()

    if filtered_catalog.empty:
        raise ValueError('Nenhum artista encontrado com os filtros informados.')

    filtered_distances = distances[candidate_mask]
    filtered_catalog['distance'] = filtered_distances
    filtered_catalog = filtered_catalog.sort_values(['distance', 'Nota', 'artist_name'], ascending=[True, False, True]).head(top_k_safe(min(top_k, len(filtered_catalog))))

    recommendation_rows = []
    for _, row in filtered_catalog.iterrows():
        score = float(100.0 * np.exp(-float(row['distance']) / 2.0))
        recommendation_rows.append((row, score))

    top_group = filtered_catalog['style_group'].mode(dropna=True)
    predicted_group = top_group.iloc[0] if not top_group.empty else filtered_catalog.iloc[0]['style_group']
    matched_score = 100.0

    return {
        'rows': recommendation_rows,
        'predicted_group': predicted_group,
        'matched_score': matched_score,
    }


def top_k_safe(candidate_count: int) -> int:
    return max(1, min(12, candidate_count))


def _artist_summary(row: pd.Series) -> dict[str, object]:
    return {
        'artist_name': row['artist_name'],
        'style': row['estilo'],
        'slug': row['SLUG'],
        'city': row['cidade'],
        'state': row['uf'],
    }


def _artist_detail(row: pd.Series) -> dict[str, object]:
    detail = _artist_summary(row)
    detail.update(
        {
            'style_group': row['style_group'],
            'formation': row['FORMACAO'],
            'song': row['video_title'],
            'note': float(row['Nota']) if pd.notna(row['Nota']) else None,
            'tempo_bpm': float(row['tempo_bpm']) if pd.notna(row['tempo_bpm']) else None,
            'video_channel': row['video_channel'],
        }
    )
    return detail


def _recommendation_detail(row: pd.Series, score: float) -> dict[str, object]:
    detail = _artist_detail(row)
    detail.update(
        {
            'score': score,
            'distance': float(row['distance']) if 'distance' in row and pd.notna(row['distance']) else None,
        }
    )
    return detail
