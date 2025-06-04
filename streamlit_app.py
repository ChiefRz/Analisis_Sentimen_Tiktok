import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import sqlite3
import os
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Sastrawi for stemming
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder # Untuk pseudo-label encoding

# Hugging Face Transformers
from transformers import pipeline as hf_pipeline # Rename to avoid conflict with sklearn.pipeline

# --- PAGE CONFIG (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(page_title="Analisis Sentimen SVM", layout="wide")

# --- NLTK Resource Downloads ---
try:
    word_tokenize("test")
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    stopwords.words('indonesian')
except LookupError:
    nltk.download('stopwords', quiet=True)

# --- Database Setup ---
DB_NAME = 'app_sentiment_data.db'
UPLOAD_DIR = "app_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

conn = sqlite3.connect(DB_NAME, check_same_thread=False)
c = conn.cursor()
c.execute('''
          CREATE TABLE IF NOT EXISTS uploaded_files (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              filename TEXT UNIQUE,
              filepath TEXT,
              original_text_column TEXT, /* To remember user's choice */
              datetime_uploaded TEXT DEFAULT CURRENT_TIMESTAMP
          )
          ''')
conn.commit()

# --- Sastrawi Stemmer Initialization ---
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# --- Hugging Face Pre-trained Model Initialization ---
@st.cache_resource # Cache the pipeline resource
def load_hf_sentiment_pipeline_core(): # Core loading logic without Streamlit elements
    try:
        sentiment_pipeline_obj = hf_pipeline( # Renamed to avoid conflict in scope
            "sentiment-analysis",
            model="w11wo/indonesian-roberta-base-sentiment-classifier",
            tokenizer="w11wo/indonesian-roberta-base-sentiment-classifier"
        )
        return sentiment_pipeline_obj, None # pipeline, error_message
    except Exception as e:
        # Return the error message instead of calling st.sidebar.error here
        return None, f"Gagal memuat model pre-trained: {e}. Beberapa fitur mungkin tidak berfungsi."

# Load the model (the actual Streamlit messages will be shown later, in the sidebar section)
sentiment_analyzer_pipeline, hf_model_load_error = load_hf_sentiment_pipeline_core()

# --- Session State Initialization ---
# (Ini bisa tetap di sini karena tidak menjalankan perintah Streamlit secara langsung)
if 'uploaded_file_info' not in st.session_state:
    st.session_state.uploaded_file_info = None
if 'selected_text_col_preprocessing' not in st.session_state:
    st.session_state.selected_text_col_preprocessing = None
if 'processed_df_info' not in st.session_state:
    st.session_state.processed_df_info = None
if 'svm_training_data' not in st.session_state:
    st.session_state.svm_training_data = None
if 'tfidf_vectorizer' not in st.session_state:
    st.session_state.tfidf_vectorizer = None
if 'svm_model' not in st.session_state:
    st.session_state.svm_model = None
if 'pseudo_label_encoder' not in st.session_state:
    st.session_state.pseudo_label_encoder = None
if 'classification_target_df_info' not in st.session_state:
    st.session_state.classification_target_df_info = None
if 'classified_df_info' not in st.session_state:
    st.session_state.classified_df_info = None

# --- Helper Functions ---
@st.cache_data(ttl=3600)
def load_df_from_path(file_path, filename_hint=""):
    try:
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path, on_bad_lines='skip')
        elif file_path.endswith('.xlsx'):
            return pd.read_excel(file_path)
        elif file_path.endswith('.txt'):
            try:
                return pd.read_csv(file_path, sep="\t", on_bad_lines='skip', header=None, names=['text'])
            except pd.errors.ParserError:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = [line.strip() for line in f if line.strip()]
                return pd.DataFrame(lines, columns=['text'])
        st.error(f"Format file tidak didukung: {file_path}") # Ini perintah Streamlit, tapi dalam fungsi yg dipanggil nanti
        return None
    except pd.errors.EmptyDataError:
        st.error(f"File '{filename_hint}' kosong.") # Ini perintah Streamlit, tapi dalam fungsi yg dipanggil nanti
        return None
    except Exception as e:
        st.error(f"Error saat membaca file '{filename_hint}': {e}") # Ini perintah Streamlit, tapi dalam fungsi yg dipanggil nanti
        return None

@st.cache_data(ttl=3600)
def _preprocess_text_cached(text, _sastrawi_stemmer_obj):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+', '', text)
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"
                           u"\U0001F300-\U0001F5FF"
                           u"\U0001F680-\U0001F6FF"
                           u"\U0001F1E0-\U0001F1FF"
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    text = text.translate(str.maketrans('', '', string.punctuation.replace("#", "")))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    stop_words_indonesian = set(stopwords.words('indonesian'))
    stemmed_tokens = [_sastrawi_stemmer_obj.stem(token) for token in tokens]
    filtered_stemmed_tokens = [word for word in stemmed_tokens if word not in stop_words_indonesian and len(word) > 1]
    return ' '.join(filtered_stemmed_tokens)

def preprocess_text_series_st(text_series, sastrawi_stemmer_obj):
    processed_texts = []
    total = len(text_series)
    if total == 0: return []
    progress_bar = st.progress(0, text="Memproses teks...") # Perintah Streamlit, tapi dalam fungsi yg dipanggil nanti
    for i, text_content in enumerate(text_series):
        processed_texts.append(_preprocess_text_cached(text_content, sastrawi_stemmer_obj))
        progress_bar.progress((i + 1) / total, text=f"Memproses teks... {i+1}/{total}")
    progress_bar.empty()
    return processed_texts

# --- Judul Aplikasi ---
# st.set_page_config sudah dipindah ke atas
st.title("📱 Aplikasi Analisis Sentimen (SVM & Pseudo-Labeling)")

# --- Pilihan Menu ---
# Sekarang kita bisa menampilkan status model pre-trained di sidebar
st.sidebar.title("Menu Navigasi")
if hf_model_load_error:
    st.sidebar.error(hf_model_load_error) # Tampilkan error di sini
elif sentiment_analyzer_pipeline:
    st.sidebar.success("Model pre-trained (sentiment) siap.") # Tampilkan success di sini

menu_options = [
    "1. Upload Data",
    "2. Preprocessing Data",
    "3. Pelatihan Model SVM (Pseudo-Labeling)",
    "4. Klasifikasi Sentimen dengan SVM",
    "5. Hasil Analisis Sentimen"
]
choice = st.sidebar.selectbox("Pilih Langkah:", menu_options)
st.markdown("---")

# --- Konten Menu ---
# (Sisa kode Anda dimulai dari sini)
if choice == "1. Upload Data":
    st.header("📤 1. Upload Data Komentar")
    uploaded_file_obj = st.file_uploader("Pilih file (CSV, XLSX, TXT):", type=['csv', 'xlsx', 'txt'])

    if uploaded_file_obj is not None:
        filename = uploaded_file_obj.name
        file_path = os.path.join(UPLOAD_DIR, filename)

        with open(file_path, "wb") as f:
            f.write(uploaded_file_obj.getbuffer())

        df = load_df_from_path(file_path, filename) # Helper function might call st.error

        if df is not None and not df.empty:
            st.success(f"File '{filename}' berhasil diunggah dan dibaca.")
            st.subheader("Pratinjau Data:")
            st.dataframe(df.head())
            st.info(f"Jumlah baris: {len(df)}, Jumlah kolom: {len(df.columns)}")

            # Auto-detect or let user select text column
            common_text_cols = ['text', 'teks', 'review', 'tweet', 'komentar', 'ulasan', 'content', 'message']
            detected_text_col = None
            df_cols_lower = [col.lower() for col in df.columns] # Get lowercased column names for matching
            
            for col_query in common_text_cols:
                if col_query in df_cols_lower: # case-insensitive check
                    # find the original actual column name
                    actual_col_name_index = df_cols_lower.index(col_query)
                    detected_text_col = df.columns[actual_col_name_index]
                    break
            
            col_options = ["Pilih kolom..."] + df.columns.tolist()
            if detected_text_col and detected_text_col in col_options:
                 default_idx = col_options.index(detected_text_col)
            else:
                 default_idx = 0

            selected_col = st.selectbox(
                "Pilih kolom yang berisi teks utama untuk analisis:",
                col_options,
                index=default_idx,
                key="upload_text_col_select"
            )

            if st.button("💾 Konfirmasi File dan Kolom Teks", key="confirm_upload_btn"):
                if selected_col != "Pilih kolom...":
                    try:
                        c.execute("INSERT INTO uploaded_files (filename, filepath, original_text_column) VALUES (?, ?, ?)",
                                  (filename, file_path, selected_col))
                        conn.commit()
                        db_id = c.lastrowid
                        st.session_state.uploaded_file_info = {'id': db_id, 'name': filename, 'path': file_path, 'df': df, 'text_col': selected_col}
                        st.success(f"File '{filename}' dengan kolom teks '{selected_col}' dikonfirmasi dan disimpan ke database (ID: {db_id}). Lanjutkan ke Preprocessing.")
                    except sqlite3.IntegrityError:
                        st.warning(f"File dengan nama '{filename}' sudah ada di database. Jika ini file yang berbeda, ubah nama file sebelum mengunggah. Jika ini file yang sama, Anda bisa lanjut ke Preprocessing jika sudah dikonfirmasi sebelumnya.")
                        c.execute("SELECT id, filepath, original_text_column FROM uploaded_files WHERE filename = ?", (filename,))
                        existing_rec = c.fetchone()
                        if existing_rec:
                            # Load existing df if not already loaded, or confirm if df is the same.
                            # For simplicity, assume df is already loaded from the uploaded file.
                            st.session_state.uploaded_file_info = {'id': existing_rec[0], 'name': filename, 'path': existing_rec[1], 'df': df, 'text_col': existing_rec[2]}
                            st.info(f"Menggunakan data file '{filename}' yang sudah ada di DB dengan kolom teks '{existing_rec[2]}'.")
                else:
                    st.error("Harap pilih kolom teks yang valid.")
        # No specific handling for "df is None" here as load_df_from_path already calls st.error
        # elif df is None:
        #     st.error("Gagal memuat data dari file. File mungkin rusak atau format tidak didukung.")

    st.subheader("📁 File Tersimpan di Database")
    try:
        db_df = pd.read_sql_query("SELECT id, filename, original_text_column, datetime_uploaded FROM uploaded_files ORDER BY datetime_uploaded DESC", conn)
        if not db_df.empty:
            st.dataframe(db_df, use_container_width=True)
        else:
            st.info("Belum ada file di database.")
    except Exception as e:
        st.error(f"Gagal memuat data dari database: {e}")
# ... (sisa kode Anda) ...