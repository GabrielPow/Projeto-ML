from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / 'src'

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ibmecPCDV20522_gabriel_pedro.models.predict import (  # noqa: E402
    build_catalog_summary,
    recommend_by_artist,
    search_artists,
)


app = FastAPI(title='Groova Recommender API', version='1.0.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


class RecommendationRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=6, ge=1, le=12)
    style_group: str | None = None
    formation: str | None = None
    bpm_min: float | None = Field(default=None, ge=0)
    bpm_max: float | None = Field(default=None, ge=0)
    min_note: float | None = Field(default=None, ge=0)


@app.get('/api/health')
def health_check() -> dict[str, str]:
    return {'status': 'ok'}


@app.get('/api/catalog')
def catalog_summary() -> dict[str, object]:
    return build_catalog_summary()


@app.get('/api/search')
def search_catalog(q: str = Query(..., min_length=1), limit: int = Query(default=6, ge=1, le=12)) -> dict[str, object]:
    return {
        'query': q,
        'matches': search_artists(q, limit=limit),
    }


@app.post('/api/recommendations')
def recommend(payload: RecommendationRequest) -> dict[str, object]:
    try:
        result = recommend_by_artist(
            payload.query,
            top_k=payload.top_k,
            style_group=payload.style_group,
            formation=payload.formation,
            bpm_min=payload.bpm_min,
            bpm_max=payload.bpm_max,
            min_note=payload.min_note,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=404,
            detail={
                'message': str(error),
                'suggestions': search_artists(payload.query, limit=payload.top_k),
            },
        ) from error

    if not result['found']:
        raise HTTPException(
            status_code=404,
            detail={
                'message': 'Artista não encontrado no catálogo.',
                'suggestions': result.get('suggestions', []),
            },
        )

    return result
