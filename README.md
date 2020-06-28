# vis-similarity

## instructions

1. (Recommended) Create venv and install package for requirements.txt file

2. Run `streamlit run app.py`

## to modify document embeddings

See/run `01-get-embeddings.ipynb`. This will use file `data/vispapers-updated.csv` and output `data/vispapers-updated-umap`, which is the input data to the streamlit app, and `data/docs_emb_updated_06272020`, which are the document vectors run in batch.

## Exploratory text analysis

See `spacy-notes.ipynb`. I used `spacy` + `textacy` packages. Easy to use if familar with their code. Open to any other packages (e.g., `gensim`) but I prefer `spacy`