# ML Project

**Repo:** ibmecPCDV20522_gabriel_pedro

**Disciplina:** Projeto de Ciência de Dados V 
**Python:** 3.12

## Equipe
**Membros:** Gabriel Pow; Pedro Pintor, Matheus Okamura  
**E-mails:** gabriel.pow2@gmail.com; pedro20pintor@gmail.com


## Estrutura do projeto
- `data/raw` : dados originais (não editar)
- `data/interim` : dados intermediários
- `data/processed` : dataset final para modelagem
- `notebooks/` : exploração/EDA (protótipos)
- `src/` : código reutilizável (pipeline)
- `models/` : artefatos treinados (pkl, joblib, etc.)
- `reports/` : relatório final + figuras/tabelas
- `configs/` : configs de experimentos (yaml/json)

## Como rodar (mínimo)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
