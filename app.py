"""
MedFAQ Chatbot — Streamlit App
Diporting dari Medical_Chatbot_535230199.ipynb

Menjalankan retrieval hybrid (TF-IDF word + char n-gram, opsional Sentence-BERT)
di atas dataset FAQ medis, lalu menampilkannya sebagai antarmuka chat.
"""

import re
from html import escape

import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------------------
# KONFIGURASI HALAMAN
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="MedFAQ — Chart-style medical FAQ retrieval",
    page_icon="🩺",
    layout="wide",
)

DATA_PATH = Path(__file__).parent / "medical_chatbot_dataset.csv"
DEFAULT_THRESHOLD = 0.15  # nilai default ambang batas skor (bisa diubah di sidebar)

INTENT_KEYWORDS = {
    "symptoms": ["symptom", "symptoms", "sign", "signs", "feel like", "indication"],
    "treatment": ["treatment", "treat", "treated", "therapy", "therapies", "medicine", "medication", "cure"],
    "causes": ["cause", "causes", "caused", "why", "reason"],
    "prevention": ["prevent", "prevention", "reduce risk", "avoid"],
    "outlook": ["outlook", "prognosis", "life expectancy", "recover", "recovery"],
    "diagnosis": ["diagnos", "exam", "exams", "test", "tests", "diagnostic"],
    "inheritance": ["inherit", "inherited", "genetic", "hereditary", "passed down"],
    "susceptibility": ["risk", "susceptible", "prone", "who is at risk"],
    "frequency": ["common", "frequent", "frequency", "how many", "how often", "prevalence"],
    "research": ["research", "clinical trial", "clinical trials", "studies"],
    "information": ["what is", "what are", "define", "definition", "information about"],
}

# ----------------------------------------------------------------------
# DESAIN — palet "chart klinis": kertas rekam medis + tinta + satu aksen
# ----------------------------------------------------------------------
PAPER = "#F2F4EF"
INK = "#1B211D"
MUTED_INK = "#5B6660"
LINE = "#D7DDD4"
ACCENT = "#A13342"     # merah klinis — dipakai untuk wordmark & fokus saja
NAVY = "#223A5E"        # entri pertanyaan pengguna
GREEN = "#3F6B52"       # entri jawaban / status normal
AMBER = "#8F5E10"       # peringatan skor rendah

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Serif:wght@400;600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {{
    --paper: {PAPER}; --ink: {INK}; --muted: {MUTED_INK}; --line: {LINE};
    --accent: {ACCENT}; --navy: {NAVY}; --green: {GREEN}; --amber: {AMBER};
}}

.stApp {{ background: var(--paper); }}
html, body, [class*="css"] {{ font-family: 'IBM Plex Sans', sans-serif; color: var(--ink); }}

.block-container {{ max-width: 900px; padding-top: 1.6rem; }}

/* ---------- Header ---------- */
.mf-header {{ border-bottom: 1px solid var(--line); padding-bottom: 14px; margin-bottom: 6px; }}
.mf-wordmark {{ font-family: 'IBM Plex Serif', serif; font-weight: 600; font-size: 30px;
                display: flex; align-items: center; gap: 10px; }}
.mf-mark {{ width: 11px; height: 11px; background: var(--accent); display: inline-block; flex: none; }}
.mf-tagline {{ font-family: 'IBM Plex Mono', monospace; font-size: 12.5px; color: var(--muted);
               margin-top: 4px; letter-spacing: .2px; }}

/* ---------- Vitals row ---------- */
.mf-vitals {{ display: flex; gap: 28px; border-top: 1px solid var(--line);
              border-bottom: 1px solid var(--line); padding: 10px 0; margin: 14px 0 18px; flex-wrap: wrap; }}
.mf-vital .label {{ font-size: 11.5px; color: var(--muted); }}
.mf-vital .value {{ font-family: 'IBM Plex Mono', monospace; font-size: 15px; font-weight: 500; }}

/* ---------- Disclaimer ---------- */
.mf-disclaimer {{ font-size: 12.5px; color: var(--muted); border-left: 2px solid var(--line);
                   padding: 4px 0 4px 10px; margin-bottom: 20px; }}

/* ---------- Example chips (st.button) ---------- */
div[data-testid="stHorizontalBlock"] .stButton > button {{
    background: transparent; border: 1px solid var(--line); border-radius: 3px;
    color: var(--ink); font-size: 13px; padding: 6px 10px; width: 100%; text-align: left;
    box-shadow: none;
}}
div[data-testid="stHorizontalBlock"] .stButton > button:hover {{ border-color: var(--navy); color: var(--navy); }}

/* ---------- Chart entries (Q/A) ---------- */
.mf-entry {{ margin-bottom: 22px; }}
.mf-row {{ display: flex; gap: 12px; padding: 9px 0 9px 12px; border-left: 3px solid transparent; }}
.mf-row.user {{ border-color: var(--navy); }}
.mf-row.bot {{ border-color: var(--green); }}
.mf-row.bot.low {{ border-color: var(--amber); }}
.mf-tag {{ font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: var(--muted);
           flex: none; padding-top: 3px; min-width: 26px; }}
.mf-body {{ flex: 1; }}
.mf-q-text {{ font-size: 15px; }}
.mf-a-text {{ font-family: 'IBM Plex Serif', serif; font-size: 15.5px; line-height: 1.65; white-space: pre-wrap; }}
.mf-rule {{ border: none; border-top: 1px dashed var(--line); margin: 8px 0 8px 38px; }}
.mf-meta {{ font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: var(--muted);
            margin: 8px 0 0 0; display: flex; gap: 16px; flex-wrap: wrap; }}
.mf-meta a {{ color: var(--navy); }}
.mf-caution {{ font-size: 13px; color: var(--amber); margin-top: 4px; }}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {{ background: #ECEFE9; border-right: 1px solid var(--line); }}

/* ---------- Chat input ---------- */
[data-testid="stChatInput"] {{ border-top: 1px solid var(--line); }}
</style>
""", unsafe_allow_html=True)


def entry_html(turn_no: int, question: str, is_low: bool, body_html: str) -> str:
    row_class = "bot low" if is_low else "bot"
    return f"""
    <div class="mf-entry">
        <div class="mf-row user">
            <div class="mf-tag">Q{turn_no}</div>
            <div class="mf-body mf-q-text">{escape(question)}</div>
        </div>
        <hr class="mf-rule"/>
        <div class="mf-row {row_class}">
            <div class="mf-tag">A{turn_no}</div>
            <div class="mf-body">{body_html}</div>
        </div>
    </div>
    """


# ----------------------------------------------------------------------
# UTIL
# ----------------------------------------------------------------------
def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9'\s-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_intent(query: str):
    q = normalize_text(query)
    return [intent for intent, words in INTENT_KEYWORDS.items() if any(term in q for term in words)]


# ----------------------------------------------------------------------
# LOAD & INDEX DATA (di-cache supaya tidak dihitung ulang setiap interaksi)
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["question", "answer", "question_type", "focus"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {missing}")

    for col in ["question", "answer", "question_type", "focus", "source", "url"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()

    df = (
        df[(df["question"] != "") & (df["answer"] != "")]
        .drop_duplicates(subset=["question", "answer"])
        .reset_index(drop=True)
    )

    df["normalized_question"] = df["question"].apply(normalize_text)
    df["normalized_focus"] = df["focus"].apply(normalize_text)
    df["normalized_type"] = df["question_type"].apply(normalize_text)
    df["search_text"] = (
        df["normalized_question"] + " " + df["normalized_focus"] + " " + df["normalized_type"]
    ).str.strip()
    return df


@st.cache_resource(show_spinner="Membangun indeks TF-IDF...")
def build_tfidf_index(search_text: tuple):
    word_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=50000, sublinear_tf=True)
    char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_features=60000, sublinear_tf=True)
    word_matrix = word_vectorizer.fit_transform(search_text)
    char_matrix = char_vectorizer.fit_transform(search_text)
    return word_vectorizer, char_vectorizer, word_matrix, char_matrix


@st.cache_resource(show_spinner="Memuat model Sentence-BERT (hanya sekali)...")
def load_sbert_model():
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return model
    except Exception:
        return None


@st.cache_resource(show_spinner="Menghitung embedding semantik...")
def build_sbert_embeddings(_model, search_text: tuple):
    if _model is None:
        return None
    return _model.encode(list(search_text), normalize_embeddings=True, show_progress_bar=False)


# ----------------------------------------------------------------------
# RETRIEVAL
# ----------------------------------------------------------------------
def retrieve(query, df, word_vectorizer, char_vectorizer, word_matrix, char_matrix,
             sbert_model, sbert_embeddings, top_k=3):
    q = normalize_text(query)
    if not q:
        return pd.DataFrame()

    q_word = word_vectorizer.transform([q])
    q_char = char_vectorizer.transform([q])
    word_scores = cosine_similarity(q_word, word_matrix).ravel()
    char_scores = cosine_similarity(q_char, char_matrix).ravel()

    use_sbert = sbert_model is not None and sbert_embeddings is not None
    if use_sbert:
        q_emb = sbert_model.encode([query], normalize_embeddings=True)
        semantic_scores = (sbert_embeddings @ q_emb[0]).ravel()
        scores = 0.60 * semantic_scores + 0.25 * word_scores + 0.15 * char_scores
    else:
        semantic_scores = np.zeros(len(df))
        scores = 0.65 * word_scores + 0.35 * char_scores

    intents = detect_intent(query)
    if intents:
        type_norm = df["normalized_type"].to_numpy()
        intent_bonus = np.array([
            1.0 if any(intent in typ or typ in intent for intent in intents) else 0.0
            for typ in type_norm
        ])
        scores = scores + 0.08 * intent_bonus

    q_compact = re.sub(r"[^a-z0-9]", "", q)
    focus_bonus = np.zeros(len(df))
    if q_compact:
        for i, focus in enumerate(df["normalized_focus"]):
            f_compact = re.sub(r"[^a-z0-9]", "", focus)
            if len(f_compact) >= 4 and f_compact in q_compact:
                focus_bonus[i] = 0.18
    scores = scores + focus_bonus

    scores = np.clip(scores, 0, 1)
    order = np.argsort(scores)[::-1][:top_k]
    result = df.iloc[order][["question", "answer", "question_type", "focus", "source", "url"]].copy()
    result.insert(0, "score", scores[order])
    return result.reset_index(drop=True)


# ----------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("**Pengaturan pencarian**")
    use_sbert_toggle = st.checkbox(
        "Sentence-BERT (semantic search)",
        value=False,
        help="Lebih akurat tapi lebih berat/lambat saat pertama kali dimuat. "
             "Jika mati, sistem memakai TF-IDF (word + character n-gram) saja.",
    )
    threshold = st.slider(
        "Ambang batas skor minimum",
        min_value=0.0, max_value=0.6, value=DEFAULT_THRESHOLD, step=0.01,
        help="Jika skor tertinggi di bawah nilai ini, chatbot mengaku tidak yakin.",
    )
    top_k = st.slider("Jumlah kandidat jawaban", 1, 5, 3)

    st.markdown("---")
    if st.button("Bersihkan riwayat"):
        st.session_state.turns = []
        st.rerun()

# ----------------------------------------------------------------------
# LOAD DATA & INDEX
# ----------------------------------------------------------------------
df = load_data(DATA_PATH)
search_text_tuple = tuple(df["search_text"].tolist())
word_vectorizer, char_vectorizer, word_matrix, char_matrix = build_tfidf_index(search_text_tuple)

sbert_model, sbert_embeddings = None, None
if use_sbert_toggle:
    sbert_model = load_sbert_model()
    if sbert_model is None:
        st.sidebar.warning("Sentence-BERT gagal dimuat, memakai fallback TF-IDF.")
    else:
        sbert_embeddings = build_sbert_embeddings(sbert_model, search_text_tuple)

# ----------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------
mode_label = "Sentence-BERT + TF-IDF" if sbert_embeddings is not None else "TF-IDF word + character"
focus_count = df["focus"].nunique()

st.markdown(f"""
<div class="mf-header">
    <div class="mf-wordmark"><span class="mf-mark"></span>MedFAQ</div>
    <div class="mf-tagline">retrieval over a medical FAQ record — not a diagnosis</div>
</div>
<div class="mf-vitals">
    <div class="mf-vital"><div class="label">records</div><div class="value">{len(df):,}</div></div>
    <div class="mf-vital"><div class="label">topics</div><div class="value">{focus_count:,}</div></div>
    <div class="mf-vital"><div class="label">mode</div><div class="value">{mode_label}</div></div>
</div>
<div class="mf-disclaimer">Informasi edukatif saja. Bukan pengganti diagnosis atau nasihat medis profesional.</div>
""", unsafe_allow_html=True)

examples = [
    "What are the symptoms of Parkinson's disease?",
    "What are the treatments for asthma?",
    "What causes breast cancer?",
    "How can stroke be prevented?",
]
cols = st.columns(4)
for i, ex in enumerate(examples):
    if cols[i].button(ex, key=f"ex_{i}"):
        st.session_state.pending_query = ex

# ----------------------------------------------------------------------
# STATE & RIWAYAT (dirender sebagai entri chart bernomor)
# ----------------------------------------------------------------------
if "turns" not in st.session_state:
    st.session_state.turns = []

history_slot = st.container()


def render_history():
    with history_slot:
        for i, t in enumerate(st.session_state.turns, start=1):
            st.markdown(entry_html(i, t["question"], t["is_low"], t["body_html"]), unsafe_allow_html=True)


render_history()

# ----------------------------------------------------------------------
# INPUT PENGGUNA
# ----------------------------------------------------------------------
user_query = st.chat_input("Tanyakan sesuatu tentang gejala, pengobatan, penyebab, dll (Bahasa Inggris)...")
if "pending_query" in st.session_state:
    user_query = st.session_state.pop("pending_query")

if user_query:
    matches = retrieve(
        user_query, df, word_vectorizer, char_vectorizer, word_matrix, char_matrix,
        sbert_model, sbert_embeddings, top_k=top_k,
    )

    if matches.empty:
        body_html = '<div class="mf-a-text">Belum ada jawaban yang cocok. Coba pertanyaan lain.</div>'
        is_low = True
    else:
        best = matches.iloc[0]
        score = float(best["score"])
        is_low = score < threshold

        if is_low:
            body_html = (
                f'<div class="mf-a-text">Belum cukup yakin dengan jawaban di dataset ini untuk pertanyaan tersebut.</div>'
                f'<div class="mf-caution">Coba tulis ulang atau sebutkan nama penyakitnya secara eksplisit.</div>'
                f'<div class="mf-meta"><span>match {score:.3f}</span></div>'
            )
        else:
            answer = escape(str(best["answer"])).replace("\n", "<br>")
            topic = escape(str(best.get("focus", "")) or "tidak diketahui")
            qtype = escape(str(best.get("question_type", "")) or "tidak diketahui")
            source = str(best.get("source", "")).strip()
            url = str(best.get("url", "")).strip()

            meta_parts = [f"<span>focus: {topic}</span>", f"<span>type: {qtype}</span>", f"<span>match: {score:.3f}</span>"]
            if source:
                meta_parts.append(f"<span>source: {escape(source)}</span>")
            if url.lower().startswith(("http://", "https://")):
                meta_parts.append(f'<span><a href="{escape(url, quote=True)}" target="_blank" rel="noopener noreferrer">open reference ↗</a></span>')

            body_html = f'<div class="mf-a-text">{answer}</div><div class="mf-meta">{"".join(meta_parts)}</div>'

            if len(matches) > 1:
                alt_items = "".join(
                    f"<li>{escape(str(row['question']))} — {float(row['score']):.3f}</li>"
                    for _, row in matches.iloc[1:].iterrows()
                )
                body_html += (
                    f'<details style="margin-top:8px;font-size:13px;color:var(--muted)">'
                    f"<summary style='cursor:pointer'>Pertanyaan lain yang cocok</summary>"
                    f"<ul>{alt_items}</ul></details>"
                )

    st.session_state.turns.append({"question": user_query, "body_html": body_html, "is_low": is_low})
    st.rerun()
