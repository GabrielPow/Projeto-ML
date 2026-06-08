import { useEffect, useMemo, useState } from 'react'
import './MusicRecommenderPage.css'

const API_BASE = import.meta.env.VITE_API_BASE_URL || '/api'
const topKOptions = [4, 6, 8, 10]

function formatScore(score) {
  if (typeof score !== 'number' || Number.isNaN(score)) {
    return '0%'
  }

  return `${Math.max(0, Math.min(100, score)).toFixed(1)}%`
}

function createInitialState() {
  return {
    query: '',
    topK: 6,
    filters: {
      styleGroup: '',
      formation: '',
      bpmMin: '',
      bpmMax: '',
      minNote: '',
    },
    isLoading: false,
    isSearching: false,
    error: '',
    catalog: null,
    suggestions: [],
    result: null,
  }
}

export default function MusicRecommenderPage() {
  const [state, setState] = useState(createInitialState)
  const quickPicks = useMemo(() => state.catalog?.top_artists ?? [], [state.catalog])
  const styleGroupOptions = state.catalog?.style_groups_options ?? []
  const formationOptions = state.catalog?.formations ?? []

  useEffect(() => {
    let isMounted = true

    async function loadCatalog() {
      try {
        const response = await fetch(`${API_BASE}/catalog`)

        if (!response.ok) {
          throw new Error('Não foi possível carregar o catálogo de artistas.')
        }

        const data = await response.json()

        if (isMounted) {
          setState((current) => ({ ...current, catalog: data, error: '' }))
        }
      } catch {
        if (isMounted) {
          setState((current) => ({
            ...current,
            error: 'O backend de predição não está disponível. Inicie a API Python para usar o modelo.',
          }))
        }
      }
    }

    loadCatalog()

    return () => {
      isMounted = false
    }
  }, [])

  useEffect(() => {
    const query = state.query.trim()

    if (query.length < 2) {
      setState((current) => ({ ...current, suggestions: [], isSearching: false }))
      return undefined
    }

    const controller = new AbortController()
    setState((current) => ({ ...current, isSearching: true }))

    const debounce = window.setTimeout(async () => {
      try {
        const response = await fetch(`${API_BASE}/search?q=${encodeURIComponent(query)}&limit=6`, {
          signal: controller.signal,
        })

        if (!response.ok) {
          throw new Error('Busca indisponível.')
        }

        const data = await response.json()
        setState((current) => ({ ...current, suggestions: data.matches ?? [], isSearching: false }))
      } catch {
        if (!controller.signal.aborted) {
          setState((current) => ({ ...current, suggestions: [], isSearching: false }))
        }
      }
    }, 250)

    return () => {
      controller.abort()
      window.clearTimeout(debounce)
    }
  }, [state.query])

  async function handleSubmit(event) {
    event.preventDefault()

    const query = state.query.trim()

    if (!query) {
      setState((current) => ({ ...current, error: 'Digite o nome de um artista ou banda.' }))
      return
    }

    setState((current) => ({ ...current, isLoading: true, error: '' }))

    try {
      const payload = {
        query,
        top_k: state.topK,
        style_group: state.filters.styleGroup || null,
        formation: state.filters.formation || null,
        bpm_min: state.filters.bpmMin === '' ? null : Number(state.filters.bpmMin),
        bpm_max: state.filters.bpmMax === '' ? null : Number(state.filters.bpmMax),
        min_note: state.filters.minNote === '' ? null : Number(state.filters.minNote),
      }

      const response = await fetch(`${API_BASE}/recommendations`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      })

      const data = await response.json()

      if (!response.ok) {
        const detail = data?.detail
        const message = typeof detail === 'string' ? detail : detail?.message || 'Nenhuma correspondência encontrada.'
        throw new Error(message)
      }

      setState((current) => ({ ...current, result: data, isLoading: false }))
    } catch (error) {
      setState((current) => ({
        ...current,
        isLoading: false,
        error: error instanceof Error ? error.message : 'Falha ao consultar o modelo.',
      }))
    }
  }

  function applyQuickPick(name) {
    setState((current) => ({ ...current, query: name, error: '' }))
  }

  function updateFilter(field, value) {
    setState((current) => ({
      ...current,
      filters: {
        ...current.filters,
        [field]: value,
      },
    }))
  }

  return (
    <main className="app-shell">
      <section className="hero-panel">
        <div className="hero-copy">
          <p className="eyebrow">Predições Groova</p>
          <h1>Descubra músicas e artistas com mais contexto</h1>
          <p className="hero-text">
            Escolha um artista de referência, refine pelos filtros musicais e veja sugestões geradas a partir do
            pipeline de dados e da similaridade em LDA.
          </p>

          <div className="hero-stats">
            <div>
              <span>Catálogo</span>
              <strong>{state.catalog?.total_artists ?? '---'}</strong>
            </div>
            <div>
              <span>Grupos musicais</span>
              <strong>{state.catalog?.style_groups ?? '---'}</strong>
            </div>
            <div>
              <span>Modelo ativo</span>
              <strong>LDA + similaridade</strong>
            </div>
          </div>
        </div>

        <aside className="hero-aside">
          <div className="status-card">
            <span className="status-label">Estado da API</span>
            <strong>{state.error && !state.catalog ? 'Offline' : 'Online'}</strong>
            <p>{state.catalog ? 'Catálogo carregado com sucesso.' : 'Aguardando resposta do backend Python.'}</p>
          </div>

          <div className="status-card subtle">
            <span className="status-label">Dado de entrada</span>
            <strong>Nome do artista</strong>
            <p>O modelo usa a linha correspondente no dataset para retornar predição e vizinhos próximos.</p>
          </div>
        </aside>
      </section>

      <section className="control-panel">
        <form className="search-form" onSubmit={handleSubmit}>
          <label>
            <span>Artista ou banda</span>
            <input
              type="text"
              value={state.query}
              onChange={(event) => setState((current) => ({ ...current, query: event.target.value }))}
              placeholder="Ex.: Marcos Rosalle Corporativo"
            />
          </label>

          <label>
            <span>Quantidade de recomendações</span>
            <select
              value={state.topK}
              onChange={(event) => setState((current) => ({ ...current, topK: Number(event.target.value) }))}
            >
              {topKOptions.map((option) => (
                <option key={option} value={option}>
                  Top {option}
                </option>
              ))}
            </select>
          </label>

          <button type="submit" disabled={state.isLoading}>
            {state.isLoading ? 'Consultando modelo...' : 'Gerar predição'}
          </button>
        </form>

        <div className="filters-grid">
          <label>
            <span>Grupo musical</span>
            <select value={state.filters.styleGroup} onChange={(event) => updateFilter('styleGroup', event.target.value)}>
              <option value="">Todos</option>
              {styleGroupOptions.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </label>

          <label>
            <span>Formação</span>
            <select value={state.filters.formation} onChange={(event) => updateFilter('formation', event.target.value)}>
              <option value="">Todos</option>
              {formationOptions.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </label>

          <label>
            <span>BPM mínimo</span>
            <input
              type="number"
              min="0"
              step="1"
              value={state.filters.bpmMin}
              onChange={(event) => updateFilter('bpmMin', event.target.value)}
              placeholder="Ex.: 90"
            />
          </label>

          <label>
            <span>BPM máximo</span>
            <input
              type="number"
              min="0"
              step="1"
              value={state.filters.bpmMax}
              onChange={(event) => updateFilter('bpmMax', event.target.value)}
              placeholder="Ex.: 140"
            />
          </label>

          <label>
            <span>Nota mínima</span>
            <input
              type="number"
              min="0"
              max="20"
              step="0.1"
              value={state.filters.minNote}
              onChange={(event) => updateFilter('minNote', event.target.value)}
              placeholder="Ex.: 10"
            />
          </label>
        </div>

        <div className="helper-row">
          <span>{state.isSearching ? 'Buscando correspondências...' : 'Sugestões rápidas com base na sua busca'}</span>
        </div>

        <div className="quick-picks">
          {quickPicks.map((artist) => (
            <button
              key={`${artist.artist_name}-${artist.slug}`}
              type="button"
              className="quick-pick"
              onClick={() => applyQuickPick(artist.artist_name)}
            >
              <strong>{artist.artist_name}</strong>
              <span>{artist.style}</span>
            </button>
          ))}
        </div>

        {state.suggestions.length > 0 && (
          <div className="suggestions-panel">
            {state.suggestions.map((suggestion) => (
              <button
                key={`${suggestion.artist_name}-${suggestion.slug}`}
                type="button"
                className="suggestion-chip"
                onClick={() => applyQuickPick(suggestion.artist_name)}
              >
                {suggestion.artist_name}
              </button>
            ))}
          </div>
        )}

        {state.error && <p className="error-banner">{state.error}</p>}
      </section>

      <section className="results-layout">
        <article className="prediction-card">
          <div className="section-heading">
            <div>
              <p className="eyebrow">Consulta de referência</p>
              <h2>Artista consultado</h2>
            </div>
            <span className="badge">LDA similarity</span>
          </div>

          {state.result ? (
            <div className="prediction-body">
              <div className="featured-match">
                <div>
                  <span className="muted">Entrada correspondente</span>
                  <h3>{state.result.matched_artist?.artist_name}</h3>
                </div>
                <div className="prediction-score">
                  <span>{formatScore(state.result.matched_score)}</span>
                  <small>compatibilidade</small>
                </div>
              </div>

              <div className="prediction-meta">
                <div>
                  <span>Grupo previsto</span>
                  <strong>{state.result.predicted_style_group}</strong>
                </div>
                <div>
                  <span>Estilo original</span>
                  <strong>{state.result.matched_artist?.style}</strong>
                </div>
                <div>
                  <span>Formação</span>
                  <strong>{state.result.matched_artist?.formation ?? '---'}</strong>
                </div>
              </div>

              <div className="feature-strip">
                <div>
                  <span>BPM</span>
                  <strong>{state.result.matched_artist?.tempo_bpm ?? '---'}</strong>
                </div>
                <div>
                  <span>Canal</span>
                  <strong>{state.result.matched_artist?.video_channel ?? '---'}</strong>
                </div>
                <div>
                  <span>Nota</span>
                  <strong>{state.result.matched_artist?.note ?? '---'}</strong>
                </div>
              </div>

              <p className="result-note">
                O artista consultado aparece apenas como referência. As recomendações abaixo já excluem esse mesmo
                artista do ranking.
              </p>
            </div>
          ) : (
            <p className="empty-state">
              Digite um artista e envie a consulta para ver a predição do modelo e os itens mais parecidos.
            </p>
          )}
        </article>

        <article className="recommendations-card">
          <div className="section-heading">
            <div>
              <p className="eyebrow">Ranking</p>
              <h2>Músicas e artistas similares</h2>
            </div>
            <span className="badge soft">Top {state.topK}</span>
          </div>

          {state.result?.recommendations?.length ? (
            <div className="recommendation-grid">
              {state.result.recommendations.map((item) => (
                <article key={`${item.artist_name}-${item.slug}`} className="recommendation-card">
                  <div className="recommendation-header">
                    <div>
                      <h3>{item.artist_name}</h3>
                      <p>{item.song}</p>
                    </div>
                    <span>{formatScore(item.score)}</span>
                  </div>

                  <p className="recommendation-description">
                    {item.city} - {item.state} · {item.style_group}
                  </p>

                  <div className="tag-row">
                    <span>{item.style}</span>
                    <span>{item.note ?? 'Sem nota'}</span>
                  </div>
                </article>
              ))}
            </div>
          ) : (
            <p className="empty-state compact">As recomendações aparecem aqui depois da consulta ao modelo.</p>
          )}
        </article>
      </section>
    </main>
  )
}
