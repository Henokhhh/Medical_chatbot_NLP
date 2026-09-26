"""
MedFAQ Chatbot — Streamlit App
Diporting dari Medical_Chatbot_535230199.ipynb

Menjalankan retrieval hybrid (TF-IDF word + char n-gram, opsional Sentence-BERT)
di atas dataset FAQ medis, lalu menampilkannya sebagai antarmuka chat.
"""

import re
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
    page_title="MedFAQ Chatbot",
    page_icon="🩺",
    layout="centered",
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
    st.header("⚙️ Pengaturan")
    use_sbert_toggle = st.checkbox(
        "Aktifkan Sentence-BERT (semantic search)",
        value=False,
        help="Lebih akurat tapi lebih berat/lambat saat load pertama kali. "
             "Jika mati, sistem memakai TF-IDF (word + character n-gram) saja.",
    )
    threshold = st.slider(
        "Ambang batas skor minimum",
        min_value=0.0, max_value=0.6, value=DEFAULT_THRESHOLD, step=0.01,
        help="Jika skor tertinggi di bawah nilai ini, chatbot akan bilang tidak menemukan jawaban relevan.",
    )
    top_k = st.slider("Jumlah kandidat jawaban ditampilkan", 1, 5, 3)

    st.markdown("---")
    if st.button("🗑️ Bersihkan riwayat chat"):
        st.session_state.messages = []
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
st.title("🩺 MedFAQ Chatbot")
st.caption(
    f"Mencari jawaban dari {len(df):,} pasangan tanya-jawab medis "
    f"({'Sentence-BERT + TF-IDF' if sbert_embeddings is not None else 'TF-IDF word + character'})."
)
st.info(
    "⚠️ Informasi edukatif saja. Chatbot ini bukan pengganti diagnosis atau nasihat medis profesional.",
    icon="⚠️",
)

with st.expander("💡 Contoh pertanyaan"):
    examples = [
        "What are the symptoms of Parkinson's disease?",
        "What are the treatments for asthma?",
        "What causes breast cancer?",
        "How can stroke be prevented?",
    ]
    cols = st.columns(2)
    for i, ex in enumerate(examples):
        if cols[i % 2].button(ex, key=f"ex_{i}"):
            st.session_state.pending_query = ex

# ----------------------------------------------------------------------
# CHAT STATE & RIWAYAT
# ----------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ----------------------------------------------------------------------
# INPUT PENGGUNA
# ----------------------------------------------------------------------
user_query = st.chat_input("Tanyakan sesuatu tentang gejala, pengobatan, penyebab, dll (Bahasa Inggris)...")
if "pending_query" in st.session_state:
    user_query = st.session_state.pop("pending_query")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    matches = retrieve(
        user_query, df, word_vectorizer, char_vectorizer, word_matrix, char_matrix,
        sbert_model, sbert_embeddings, top_k=top_k,
    )

    with st.chat_message("assistant"):
        if matches.empty:
            reply = "Maaf, saya tidak menemukan jawaban. Coba pertanyaan lain."
            st.markdown(reply)
        else:
            best = matches.iloc[0]
            score = float(best["score"])
            if score < threshold:
                reply = (
                    f"Saya tidak menemukan jawaban yang cukup relevan di dataset "
                    f"(skor tertinggi: {score:.3f}). Coba tulis ulang pertanyaan atau sebutkan nama penyakitnya."
                )
                st.markdown(reply)
            else:
                answer = str(best["answer"])
                topic = best.get("focus", "") or "Tidak diketahui"
                qtype = best.get("question_type", "") or "Tidak diketahui"
                source = str(best.get("source", "")).strip()
                url = str(best.get("url", "")).strip()

                reply_lines = [answer, ""]
                reply_lines.append(f"📌 **Topik:** {topic} &nbsp;|&nbsp; 🏷️ **Jenis:** {qtype} &nbsp;|&nbsp; 🎯 **Skor:** {score:.3f}")
                if source:
                    reply_lines.append(f"📚 **Sumber:** {source}")
                if url.lower().startswith(("http://", "https://")):
                    reply_lines.append(f"🔗 [Buka referensi sumber]({url})")

                reply = "\n\n".join(reply_lines)
                st.markdown(reply)

                if len(matches) > 1:
                    with st.expander("Lihat pertanyaan lain yang cocok"):
                        for _, row in matches.iloc[1:].iterrows():
                            st.markdown(f"- {row['question']}  (skor: {float(row['score']):.3f})")

    st.session_state.messages.append({"role": "assistant", "content": reply})
