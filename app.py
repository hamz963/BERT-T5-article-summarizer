import os
import re
import time
import tempfile
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from rouge import Rouge
import streamlit as st
from PyPDF2 import PdfReader


@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with st.spinner("Loading BERT model (bert-base-uncased)..."):
        bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        bert_model = BertModel.from_pretrained("bert-base-uncased").to(device).eval()

    with st.spinner("Loading T5 model (t5-small)..."):
        t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
        t5_model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device).eval()

    return bert_model, bert_tokenizer, t5_model, t5_tokenizer, device


def split_into_sentences(text: str):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def compute_sentence_embeddings(sentences, model, tokenizer, device="cpu"):
    encoded = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**encoded)
        hidden_states = outputs.last_hidden_state
        attention_mask = encoded.attention_mask.unsqueeze(-1)
        masked_hidden = hidden_states * attention_mask
        sentence_embeddings = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1e-9)
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    return sentence_embeddings.cpu()


def generate_extractive_summary(text: str, model, tokenizer, max_sentences: int = 5, device: str = "cpu") -> str:
    sentences = split_into_sentences(text)
    if not sentences:
        return ""

    sentence_embeddings = compute_sentence_embeddings(sentences, model, tokenizer, device=device)
    doc_embedding = F.normalize(sentence_embeddings.mean(dim=0, keepdim=True), p=2, dim=1)
    scores = torch.matmul(sentence_embeddings, doc_embedding.T).squeeze(-1)
    top_indices = scores.argsort(descending=True)[: min(max_sentences, len(sentences))]
    top_indices = sorted(top_indices.tolist())
    return " ".join(sentences[i] for i in top_indices)


def generate_summary(text: str, model, tokenizer, prefix: str = "", max_length: int = 130, device: str = "cpu") -> str:
    input_text = prefix + text
    tokenized = tokenizer(
        [input_text], return_tensors="pt", padding=True, truncation=True, max_length=512
    ).to(device)

    with torch.no_grad():
        summary_ids = model.generate(
            **tokenized,
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
            length_penalty=2.0,
        )

    return tokenizer.batch_decode(summary_ids, skip_special_tokens=True)[0]


def create_ensemble_summary(extractive_summary, t5_summary):
    return f"{extractive_summary} {t5_summary}"


def evaluate_rouge(generated, references):
    rouge = Rouge()
    try:
        scores = rouge.get_scores(generated, references, avg=True)
        return {
            "rouge-1": scores["rouge-1"]["f"],
            "rouge-2": scores["rouge-2"]["f"],
            "rouge-l": scores["rouge-l"]["f"],
        }
    except Exception as e:
        st.error(f"ROUGE evaluation error: {e}")
        return {"rouge-1": 0.0, "rouge-2": 0.0, "rouge-l": 0.0}


def extract_text_from_pdf(pdf_file) -> str:
    text = ""
    reader = PdfReader(pdf_file)
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()


st.set_page_config(
    page_title="Smart Summarizer (BERT + T5)",
    page_icon="Σ",
    layout="wide",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #7c3aed, #22c55e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .summary-box {
        background: #1e1e2e;
        border: 1px solid #333;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .summary-label {
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #a78bfa;
        margin-bottom: 8px;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(124,58,237,0.15), rgba(34,197,94,0.10));
        border: 1px solid rgba(124,58,237,0.3);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #22c55e;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stTextArea textarea {
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">Σ Smart Summarizer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">BERT + T5 • Extractive + Abstractive • ROUGE Evaluation</div>', unsafe_allow_html=True)

bert_model, bert_tokenizer, t5_model, t5_tokenizer, device = load_models()

tab1, tab2, tab3 = st.tabs(["Single Article", "Batch Processing", "Results"])

with tab1:
    st.header("Summarize a Single Article")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader("Upload PDF or TXT file", type=["pdf", "txt"])
        article_text = st.text_area("Or paste article text below", placeholder="Paste your article text here...", height=250)

    with col2:
        max_length = st.slider("Max summary length", min_value=50, max_value=300, value=130, step=10)
        max_sentences = st.slider("Max extractive sentences", min_value=3, max_value=10, value=5, step=1)

    if st.button("Generate Summary", type="primary", use_container_width=True):
        text_to_summarize = ""

        if uploaded_file is not None:
            if uploaded_file.name.endswith(".pdf"):
                text_to_summarize = extract_text_from_pdf(uploaded_file)
            else:
                text_to_summarize = uploaded_file.read().decode("utf-8")
        elif article_text.strip():
            text_to_summarize = article_text.strip()
        else:
            st.warning("Please upload a file or paste text to summarize.")

        if text_to_summarize:
            start_time = time.time()

            with st.spinner("Generating BERT extractive summary..."):
                bert_summary = generate_extractive_summary(
                    text_to_summarize, bert_model, bert_tokenizer, max_sentences=max_sentences, device=device
                )

            with st.spinner("Generating T5 abstractive summary..."):
                t5_summary = generate_summary(
                    bert_summary, t5_model, t5_tokenizer, prefix="summarize: ", device=device, max_length=max_length
                )

            ensemble_summary = create_ensemble_summary(bert_summary, t5_summary)
            elapsed = time.time() - start_time

            st.divider()
            st.subheader("Generated Summaries")

            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown(f'<div class="summary-box"><div class="summary-label">BERT Extractive Summary</div>{bert_summary}</div>', unsafe_allow_html=True)

            with col_b:
                st.markdown(f'<div class="summary-box"><div class="summary-label">T5 Abstractive Summary</div>{t5_summary}</div>', unsafe_allow_html=True)

            st.markdown(f'<div class="summary-box"><div class="summary-label">Ensemble Summary (Combined)</div>{ensemble_summary}</div>', unsafe_allow_html=True)

            st.divider()
            st.subheader("Statistics")

            compression = (1 - len(ensemble_summary) / len(text_to_summarize)) * 100 if len(text_to_summarize) > 0 else 0

            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{elapsed:.1f}s</div><div class="metric-label">Processing Time</div></div>', unsafe_allow_html=True)
            with col_m2:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{len(text_to_summarize)}</div><div class="metric-label">Original Chars</div></div>', unsafe_allow_html=True)
            with col_m3:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{len(ensemble_summary)}</div><div class="metric-label">Summary Chars</div></div>', unsafe_allow_html=True)
            with col_m4:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{compression:.1f}%</div><div class="metric-label">Compression</div></div>', unsafe_allow_html=True)

            results_df = pd.DataFrame({
                "Model": ["BERT Extractive", "T5 Abstractive", "Ensemble"],
                "Summary": [bert_summary, t5_summary, ensemble_summary],
            })

            os.makedirs("output", exist_ok=True)
            csv_output = os.path.join("output", "single_summary_results.csv")
            results_df.to_csv(csv_output, index=False)
            st.success(f"Results saved to {csv_output}")

with tab2:
    st.header("Batch Processing (CSV Dataset)")

    st.markdown("Process multiple articles from a CSV dataset. Expects columns: `article` and `highlights`.")

    col1, col2 = st.columns(2)
    with col1:
        sample_size = st.slider("Sample size", min_value=1, max_value=50, value=10, step=1)
    with col2:
        batch_size = st.slider("Batch size for T5", min_value=1, max_value=16, value=4, step=1)

    if st.button("Run Batch Pipeline", type="primary", use_container_width=True):
        if not os.path.exists("train_dataset.csv"):
            st.error("train_dataset.csv not found in the project directory.")
        else:
            start_time = time.time()
            os.makedirs("output", exist_ok=True)

            df = pd.read_csv("train_dataset.csv")
            st.info(f"Loaded dataset: {len(df)} total articles. Using sample of {sample_size}.")

            if sample_size < len(df):
                df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

            articles = df["article"].tolist()
            references = df["highlights"].tolist()

            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("Generating BERT extractive summaries...")
            bert_summaries = []
            for i, article in enumerate(articles):
                bert_sum = generate_extractive_summary(article, bert_model, bert_tokenizer, device=device)
                bert_summaries.append(bert_sum)
                progress_bar.progress((i + 1) / (len(articles) * 2))

            status_text.text("Generating T5 abstractive summaries...")
            t5_summaries = []
            for i in range(0, len(bert_summaries), batch_size):
                batch = bert_summaries[i:i + batch_size]
                for bert_sum in batch:
                    t5_sum = generate_summary(bert_sum, t5_model, t5_tokenizer, prefix="summarize: ", device=device)
                    t5_summaries.append(t5_sum)
                progress_bar.progress((len(bert_summaries) + i + len(batch)) / (len(articles) * 2))

            ensemble_summaries = [
                create_ensemble_summary(b, t) for b, t in zip(bert_summaries, t5_summaries)
            ]

            status_text.text("Evaluating ROUGE scores...")
            rouge_scores = evaluate_rouge(ensemble_summaries, references)

            elapsed = time.time() - start_time

            st.divider()
            st.subheader("ROUGE Scores")

            col_r1, col_r2, col_r3 = st.columns(3)
            with col_r1:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{rouge_scores["rouge-1"]:.4f}</div><div class="metric-label">ROUGE-1</div></div>', unsafe_allow_html=True)
            with col_r2:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{rouge_scores["rouge-2"]:.4f}</div><div class="metric-label">ROUGE-2</div></div>', unsafe_allow_html=True)
            with col_r3:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{rouge_scores["rouge-l"]:.4f}</div><div class="metric-label">ROUGE-L</div></div>', unsafe_allow_html=True)

            results_df = pd.DataFrame({
                "article": articles,
                "reference_summary": references,
                "bert_extractive_summary": bert_summaries,
                "t5_summary": t5_summaries,
                "ensemble_summary": ensemble_summaries,
            })

            csv_output = os.path.join("output", "summarization_results.csv")
            results_df.to_csv(csv_output, index=False)

            try:
                excel_output = os.path.join("output", "summarization_results.xlsx")
                results_df.to_excel(excel_output, index=False)
            except Exception:
                pass

            st.success(f"Pipeline completed in {elapsed:.1f}s. Results saved to {csv_output}")

            st.subheader("Sample Results")
            for i in range(min(3, len(results_df))):
                row = results_df.iloc[i]
                with st.expander(f"Article {i+1}"):
                    st.markdown(f"**Original (truncated):** {str(row['article'])[:200]}...")
                    st.markdown(f"**Reference:** {row['reference_summary']}")
                    st.markdown(f"**Ensemble:** {row['ensemble_summary']}")

with tab3:
    st.header("View Results")

    csv_path = "output/summarization_results.csv"
    single_path = "output/single_summary_results.csv"

    if os.path.exists(csv_path):
        st.subheader("Batch Results")
        results_df = pd.read_csv(csv_path)
        st.dataframe(results_df, use_container_width=True)

        if st.button("Download CSV"):
            csv_data = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download batch results as CSV",
                data=csv_data,
                file_name="summarization_results.csv",
                mime="text/csv",
            )

    if os.path.exists(single_path):
        st.subheader("Single Article Results")
        single_df = pd.read_csv(single_path)
        st.dataframe(single_df, use_container_width=True)
