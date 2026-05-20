# Smart Summarizer (BERT + T5)

An end-to-end article summarization web application that combines:
- **BERT** for extractive sentence selection
- **T5** for abstractive summary generation
- **ROUGE** for quantitative evaluation
- **Streamlit** for an interactive web UI

The app extracts key sentences using BERT embeddings, summarizes them with T5, and combines extractive and abstractive outputs into a practical ensemble result — all through a clean, modern web interface.

## Features

- **Single Article Summarization** — paste text or upload PDF/TXT files
- **Batch Processing** — process CSV datasets with progress tracking
- **ROUGE Evaluation** — automatic ROUGE-1, ROUGE-2, ROUGE-L scoring
- **Results Dashboard** — view and download summarization results
- **Statistics** — processing time, compression ratio, character counts

## What's in this repo

- `app.py` – Streamlit web application (merged frontend + backend)
- `summarize.ipynb` – Jupyter notebook pipeline (single + batch summarization + ROUGE evaluation)
- `train_dataset.csv` – dataset in CNN/DailyMail-like format
- `index.html` – static HTML frontend (legacy)
- `requirements.txt` – Python dependencies
- `README.md`

## Quick start

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Run the Streamlit app

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`.

### 3) (Optional) Run the Jupyter notebook

```bash
jupyter notebook summarize.ipynb
```

### Expected inputs / outputs

**Single Article Mode:**
- Input: paste text or upload PDF/TXT
- Output: BERT extractive, T5 abstractive, and ensemble summaries

**Batch Processing Mode:**
- Input: `train_dataset.csv` with `article` and `highlights` columns
- Output: `output/summarization_results.csv` and `.xlsx`

## How it works

1. **Load models**
   - `bert-base-uncased` for BERT extractive summarization
   - `t5-small` for T5 abstractive generation
2. **Generate summaries**
   - BERT extracts top sentences based on embedding similarity
   - T5 generates abstractive summary from extracted sentences
3. **Ensemble**
   - Concatenates extractive and abstractive outputs
4. **Evaluate**
   - Computes ROUGE-1, ROUGE-2, and ROUGE-L vs. reference summaries

## Dataset

`train_dataset.csv` includes:
- `article` – input article text
- `highlights` – reference summary

## Performance

Typical ROUGE scores on CNN/DailyMail sample:
| Metric | Score |
|--------|-------|
| ROUGE-1 | ~0.20 |
| ROUGE-2 | ~0.05 |
| ROUGE-L | ~0.19 |

## Notes

- Summarization is compute-heavy; GPU recommended for faster processing
- Models are cached on first load (subsequent runs are faster)
- The ensemble strategy concatenates extractive and abstractive summaries

## Requirements

See `requirements.txt`. Core dependencies:
- `torch`
- `transformers`
- `pandas`
- `rouge`
- `streamlit`
- `spacy`
- `PyPDF2`
