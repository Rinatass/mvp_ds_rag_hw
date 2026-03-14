# mvp_ds_rag_hw

Use poetry:

```bash
poetry install
```

If `poetry install` fails on **torch** (e.g. "wheels were skipped" on macOS), install those deps with pip inside the venv, then run the app:

```bash
poetry run pip install 'numpy<2' torch 'sentence-transformers>=2.2,<4' 'transformers<4.41'
poetry run python main.py
```