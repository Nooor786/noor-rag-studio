# Nexus RAG Studio — Hybrid Search + Streamlit

An advanced, local-first RAG interface for querying PDF documents. Includes hybrid BM25 + semantic retrieval, optional cross-encoder reranking, page-aware source citations, a chat workspace, a retrieval evaluation lab, and multi-PDF upload.

## Features

- Upload and query one or multiple PDF documents.
- Uses BM25 keyword retrieval plus Sentence Transformers semantic similarity.
- Combines normalized lexical and semantic scores (45% BM25, 55% semantic).
- Optional CrossEncoder second-stage reranking.
- Answers generated through local Ollama, default model `llama3.2:3b`.
- Source IDs, file names, page numbers, ranking scores, and original passages are displayed with each answer.
- Retrieval quality lab to test multiple questions and inspect top-k passages.
- Documents & index tab to inspect extracted documents and chunks.
- Streamlit resource caching for the embedding model and reranker.
- PDF bytes are processed in the Streamlit session; this app does not send uploaded files to a hosted vector database.

## Requirements

- Python 3.11 recommended (matches the development environment used for this project).
- Ollama installed and running locally.
- An Ollama model such as `llama3.2:3b`.

## Setup on macOS / VS Code

Open a terminal in this folder, then run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If Ollama is installed but the model is not downloaded:

```bash
ollama pull llama3.2:3b
```

Start Ollama if it is not already running:

```bash
ollama serve
```

Leave that terminal running. In another terminal, activate `.venv` and start the app:

```bash
streamlit run app.py
```

Streamlit prints a local URL, normally `http://localhost:8501`.

## Use the app

1. Upload one or more PDFs in the sidebar. Text-based PDFs are supported; scanned image PDFs need OCR.
2. Choose chunk size, overlap, number of sources, and whether to use cross-encoder reranking.
3. Ask a question in Chat workspace. Each answer includes expandable source evidence.
4. Open Retrieval quality lab and run the default set of test questions or enter your own questions, one per line.
5. Open Documents & index to preview extracted chunks.

## Existing project PDF

If `documents/sample.pdf` exists inside this folder, the app can include it using the sidebar checkbox. To use that feature, copy your existing `documents/sample.pdf` into this project's `documents/` folder. Alternatively, upload the PDF directly in the sidebar.

## Notes and limitations

- Retrieval scores are ranking signals, not probabilities or proof of correctness.
- The evaluation tab is a manual inspection harness; for formal evaluation, prepare question/expected-source labels and measure Recall@k, MRR, or nDCG.
- Scanned PDFs without a text layer will need an OCR stage.
- The embedding and reranker models may download on first use. The reranker can be disabled in the sidebar to reduce startup/download time.
- Uploaded PDFs are held in the running Streamlit session's memory; this prototype does not persist uploads to a database.
