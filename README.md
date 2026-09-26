# MedFAQ Chatbot (Streamlit)

Chatbot pencarian FAQ medis berbasis TF-IDF (dan opsional Sentence-BERT), diporting dari notebook `Medical_Chatbot_535230199.ipynb`.

## Menjalankan secara lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

Buka browser ke `http://localhost:8501`.

## File

- `app.py` — aplikasi Streamlit
- `medical_chatbot_dataset.csv` — dataset FAQ medis (wajib berada di folder yang sama dengan `app.py`)
- `requirements.txt` — daftar dependency

## Catatan

- Aktifkan toggle **Sentence-BERT** di sidebar untuk pencarian semantik yang lebih akurat (unduhan model ~90MB saat pertama kali dijalankan). Jika dimatikan, sistem memakai TF-IDF (word + character n-gram) saja — lebih ringan dan cepat.
- Chatbot ini hanya untuk tujuan edukasi, bukan pengganti diagnosis medis profesional.
