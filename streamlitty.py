nlp = spacy.load("en_core_web_sm")
from io import BytesIO
import os
import fitz  # PyMuPDF
import spacy
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from keybert import KeyBERT
import streamlit as st
import spacy.cli
spacy.cli.download("en_core_web_sm")

# ------------------ Helper Functions ------------------


@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_sm", disable=["ner", "parser"])


@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    return "\n".join(page.get_text() for page in doc)


def extract_keybert_keywords(text, top_n=15):
    return [kw for kw, _ in KeyBERT().extract_keywords(text, top_n=top_n, stop_words='english')]


def lemmatize_phrase(phrase, nlp):
    return " ".join(tok.lemma_ for tok in nlp(phrase.lower()) if tok.is_alpha)


def dedup_keywords_by_lemma(keywords, nlp):
    seen, result = {}, []
    for kw in keywords:
        lemma = lemmatize_phrase(kw, nlp)
        if lemma and lemma not in seen:
            seen[lemma] = kw
            result.append(kw)
    return result


def compute_filtered_similarity(kb_keywords, manual_keywords, model, threshold=0.65):
    sim = util.cos_sim(
        model.encode(kb_keywords, convert_to_tensor=True),
        model.encode(manual_keywords, convert_to_tensor=True)
    )
    df = pd.DataFrame(sim.cpu().numpy(), index=kb_keywords,
                      columns=manual_keywords)
    return df[df.max(axis=1) > threshold]


def sanitize_sheet_name(name):
    return "".join(c for c in name if c.isalnum() or c in " _-")[:31]


def generate_excel(output_dict):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, df in output_dict.items():
            df.to_excel(writer, sheet_name=sheet_name)
    output.seek(0)
    return output

# ------------------ Streamlit App ------------------


st.title("📄 KeyBERT Keyword Extractor for Multiple PDFs")
st.write("Upload multiple PDFs, enter keywords for each, and extract relevant terms using KeyBERT with cosine similarity filtering.")

uploaded_files = st.file_uploader(
    "Upload PDF files", type="pdf", accept_multiple_files=True)

threshold = st.slider("Cosine similarity threshold", 0.0, 1.0, 0.65, step=0.05)
top_n = st.number_input(
    "Number of KeyBERT keywords to extract", min_value=5, max_value=30, value=15)

if uploaded_files:
    nlp = load_spacy()
    model = load_model()

    pdf_keyword_inputs = {}
    for file in uploaded_files:
        with st.expander(f"Manual Keywords for: {file.name}"):
            keyword_input = st.text_area(
                "Enter manual keywords (comma-separated)", key=file.name)
            manual_keywords = [kw.strip()
                               for kw in keyword_input.split(",") if kw.strip()]
            pdf_keyword_inputs[file] = manual_keywords

    if st.button("🔍 Process PDFs"):
        results = {}

        for file, manual_keywords in pdf_keyword_inputs.items():
            text = extract_text_from_pdf(file)
            kb_raw = extract_keybert_keywords(text, top_n=top_n)
            kb_unique = dedup_keywords_by_lemma(kb_raw, nlp)
            manual_unique = dedup_keywords_by_lemma(manual_keywords, nlp)

            st.write(f"✅ Processing: `{file.name}`")
            st.write("🔍 KeyBERT keywords (deduplicated):", kb_unique)
            st.write("📝 Manual keywords (deduplicated):", manual_unique)

            df = compute_filtered_similarity(
                kb_unique, manual_unique, model, threshold)
            results[sanitize_sheet_name(file.name)] = df

        if results:
            excel_data = generate_excel(results)
            st.success("✅ Done! Download your Excel file below:")
            st.download_button("📥 Download Results as Excel",
                               data=excel_data, file_name="keybert_keywords.xlsx")
