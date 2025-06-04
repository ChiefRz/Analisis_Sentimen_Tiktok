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
from sklearn.model_selection import train_test_split # Kept for potential future use, not directly in pseudo-label path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Hugging Face Transformers
from transformers import pipeline as hf_pipeline

# --- PAGE CONFIG (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(page_title="Analisis Sentimen SVM Sederhana", layout="wide")

# --- NLTK Resource Downloads ---
@st.cache_resource
def download_nltk_resources():
    try:
        word_tokenize("test")
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        stopwords.words('indonesian')
    except LookupError:
        nltk.download('stopwords', quiet=True)
download_nltk_resources()

# --- Database Setup ---
DB_NAME = 'app_sentiment_data_simplified.db'
UPLOAD_DIR = "app_uploads_simplified"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@st.cache_resource
def get_db_connection():
    conn = sqlite3.connect(DB_NAME, check_same_thread=False)
    return conn

conn = get_db_connection()
c = conn.cursor()
c.execute('''
    CREATE TABLE IF NOT EXISTS uploaded_files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT UNIQUE,
        filepath TEXT,
        original_text_column TEXT,
        datetime_uploaded TEXT DEFAULT CURRENT_TIMESTAMP
    )
''')
conn.commit()

# --- Sastrawi Stemmer Initialization ---
@st.cache_resource
def load_sastrawi_stemmer():
    factory = StemmerFactory()
    return factory.create_stemmer()
stemmer = load_sastrawi_stemmer()

# --- Hugging Face Pre-trained Model Initialization ---
@st.cache_resource
def load_hf_sentiment_pipeline():
    try:
        sentiment_pipeline_obj = hf_pipeline(
            "sentiment-analysis",
            model="w11wo/indonesian-roberta-base-sentiment-classifier",
            tokenizer="w11wo/indonesian-roberta-base-sentiment-classifier"
        )
        return sentiment_pipeline_obj, None
    except Exception as e:
        return None, f"Gagal memuat model pre-trained: {e}. Fitur pseudo-labeling & SVM mungkin tidak optimal."
sentiment_analyzer_pipeline, hf_model_load_error = load_hf_sentiment_pipeline()

# --- Session State Initialization ---
if 'data' not in st.session_state:
    st.session_state.data = {
        'raw_df': None,
        'raw_df_path': None,
        'raw_df_filename': None,
        'raw_df_text_col': None,
        'processed_df': None,
        'classified_df': None,
    }
if 'tfidf_vectorizer' not in st.session_state:
    st.session_state.tfidf_vectorizer = None
if 'svm_model' not in st.session_state:
    st.session_state.svm_model = None
if 'pseudo_label_encoder' not in st.session_state:
    st.session_state.pseudo_label_encoder = None
if 'svm_training_details' not in st.session_state:
    st.session_state.svm_training_details = None

# --- Helper Functions ---
@st.cache_data(ttl=3600)
def load_df_from_path(_file_path, _filename_hint=""): # Underscore to indicate internal use of st
    try:
        if _file_path.endswith('.csv'):
            return pd.read_csv(_file_path, on_bad_lines='skip')
        elif _file_path.endswith('.xlsx'):
            return pd.read_excel(_file_path)
        elif _file_path.endswith('.txt'):
            try: # Try reading as TSV first
                return pd.read_csv(_file_path, sep="\t", on_bad_lines='skip', header=None, names=['text'])
            except pd.errors.ParserError: # Fallback to line-by-line
                with open(_file_path, 'r', encoding='utf-8') as f:
                    lines = [line.strip() for line in f if line.strip()]
                return pd.DataFrame(lines, columns=['text'])
        st.error(f"Format file tidak didukung: {_file_path}")
        return None
    except pd.errors.EmptyDataError:
        st.error(f"File '{_filename_hint}' kosong.")
        return None
    except Exception as e:
        st.error(f"Error saat membaca file '{_filename_hint}': {e}")
        return None

@st.cache_data(ttl=3600)
def _preprocess_text_cached(text, _sastrawi_stemmer): # Underscore to indicate internal use
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+', '', text)
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F" u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF" u"\U0001F1E0-\U0001F1FF"
                               u"\U00002702-\U000027B0" u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    text = text.translate(str.maketrans('', '', string.punctuation.replace("#", ""))) # Keep hashtags if desired
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    stop_words_indonesian = set(stopwords.words('indonesian'))
    # Use the passed stemmer object
    stemmed_tokens = [_sastrawi_stemmer.stem(token) for token in tokens]
    filtered_stemmed_tokens = [word for word in stemmed_tokens if word not in stop_words_indonesian and len(word) > 1]
    return ' '.join(filtered_stemmed_tokens)

def preprocess_text_series_st(text_series, sastrawi_stemmer_obj):
    processed_texts = []
    total = len(text_series)
    if total == 0: return pd.Series([], dtype=str) # Return empty Series for safety
    progress_bar = st.progress(0, text="Memproses teks...")
    for i, text_content in enumerate(text_series):
        processed_texts.append(_preprocess_text_cached(text_content, sastrawi_stemmer_obj)) # Pass stemmer
        progress_bar.progress((i + 1) / total, text=f"Memproses teks... {i+1}/{total}")
    progress_bar.empty()
    return pd.Series(processed_texts)


def display_df_preview(df, title="Pratinjau Data:", max_rows=5):
    if df is not None and not df.empty:
        st.subheader(title)
        st.dataframe(df.head(max_rows))
        st.info(f"Jumlah baris: {len(df)}, Jumlah kolom: {len(df.columns)}")
    elif df is not None and df.empty:
        st.info("DataFrame kosong.")

# --- Judul Aplikasi ---
st.title("📱 Aplikasi Analisis Sentimen (SVM & Pseudo-Labeling) - Versi Sederhana")

# --- Pilihan Menu ---
st.sidebar.title("Menu Navigasi")
if hf_model_load_error:
    st.sidebar.error(hf_model_load_error)
elif sentiment_analyzer_pipeline:
    st.sidebar.success("Model pre-trained (sentiment) siap.")

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

if choice == "1. Upload Data":
    st.header("📤 1. Upload Data Komentar")
    uploaded_file_obj = st.file_uploader("Pilih file (CSV, XLSX, TXT):", type=['csv', 'xlsx', 'txt'], key="file_uploader_main")

    if uploaded_file_obj is not None:
        filename = uploaded_file_obj.name
        file_path = os.path.join(UPLOAD_DIR, filename)

        with open(file_path, "wb") as f:
            f.write(uploaded_file_obj.getbuffer())

        df = load_df_from_path(file_path, filename)

        if df is not None and not df.empty:
            st.success(f"File '{filename}' berhasil diunggah dan dibaca.")
            display_df_preview(df, "Pratinjau Data Awal:")

            common_text_cols = ['text', 'teks', 'review', 'tweet', 'komentar', 'ulasan', 'content', 'message']
            df_cols_lower = [col.lower() for col in df.columns]
            detected_text_col = None
            for col_query in common_text_cols:
                if col_query in df_cols_lower:
                    actual_col_name_index = df_cols_lower.index(col_query)
                    detected_text_col = df.columns[actual_col_name_index]
                    break
            
            col_options = ["Pilih kolom..."] + df.columns.tolist()
            default_idx = col_options.index(detected_text_col) if detected_text_col and detected_text_col in col_options else 0
            selected_col = st.selectbox(
                "Pilih kolom yang berisi teks utama untuk analisis:",
                col_options, index=default_idx, key="upload_text_col_select"
            )

            if st.button("💾 Konfirmasi File dan Kolom Teks", key="confirm_upload_btn"):
                if selected_col != "Pilih kolom...":
                    try:
                        c.execute("INSERT INTO uploaded_files (filename, filepath, original_text_column) VALUES (?, ?, ?)",
                                  (filename, file_path, selected_col))
                        conn.commit()
                        db_id = c.lastrowid
                        
                        # Update session state
                        st.session_state.data = { # Reset relevant parts of data state
                            'raw_df': df,
                            'raw_df_path': file_path,
                            'raw_df_filename': filename,
                            'raw_df_text_col': selected_col,
                            'processed_df': None, # Clear previous processed/classified data
                            'classified_df': None,
                        }
                        # Also clear model artifacts as they are specific to a dataset
                        st.session_state.tfidf_vectorizer = None
                        st.session_state.svm_model = None
                        st.session_state.pseudo_label_encoder = None
                        st.session_state.svm_training_details = None


                        st.success(f"File '{filename}' (kolom: '{selected_col}') dikonfirmasi (ID DB: {db_id}). Lanjutkan ke Preprocessing.")
                    except sqlite3.IntegrityError:
                        st.warning(f"File '{filename}' sudah ada di database. Untuk memproses ulang dengan kolom ini, cukup konfirmasi.")
                        # Load existing record info and update session state if user wants to proceed with this file
                        c.execute("SELECT id, filepath, original_text_column FROM uploaded_files WHERE filename = ?", (filename,))
                        existing_rec = c.fetchone()
                        if existing_rec:
                            st.session_state.data = {
                                'raw_df': df, # Use the newly loaded df
                                'raw_df_path': existing_rec[1],
                                'raw_df_filename': filename,
                                'raw_df_text_col': selected_col, # Use current selection
                                'processed_df': None,
                                'classified_df': None,
                            }
                            st.session_state.tfidf_vectorizer = None
                            st.session_state.svm_model = None
                            st.session_state.pseudo_label_encoder = None
                            st.session_state.svm_training_details = None
                            st.info(f"Menggunakan data file '{filename}' dengan kolom teks '{selected_col}'.")
                else:
                    st.error("Harap pilih kolom teks yang valid.")

    st.subheader("📁 File Tersimpan di Database")
    try:
        db_df = pd.read_sql_query("SELECT id, filename, original_text_column, datetime_uploaded FROM uploaded_files ORDER BY datetime_uploaded DESC", conn)
        if not db_df.empty:
            st.dataframe(db_df, use_container_width=True)
        else:
            st.info("Belum ada file di database.")
    except Exception as e:
        st.error(f"Gagal memuat data dari database: {e}")


elif choice == "2. Preprocessing Data":
    st.header("🧹 2. Preprocessing Data Teks")

    source_choice = st.radio(
        "Pilih sumber data untuk preprocessing:",
        ("Gunakan data yang baru diunggah/dikonfirmasi", "Pilih dari database"),
        key="preprocess_source_choice",
        horizontal=True
    )

    df_to_process = None
    text_col_to_process = None
    filename_for_processing = None

    if source_choice == "Gunakan data yang baru diunggah/dikonfirmasi":
        if st.session_state.data['raw_df'] is not None:
            df_to_process = st.session_state.data['raw_df']
            text_col_to_process = st.session_state.data['raw_df_text_col']
            filename_for_processing = st.session_state.data['raw_df_filename']
            st.info(f"Akan memproses: '{filename_for_processing}' dengan kolom teks '{text_col_to_process}'.")
            display_df_preview(df_to_process, "Data yang akan diproses:")
        else:
            st.warning("Tidak ada data yang baru diunggah/dikonfirmasi. Silakan unggah di Langkah 1 atau pilih dari database.")

    elif source_choice == "Pilih dari database":
        try:
            c.execute("SELECT id, filename, filepath, original_text_column FROM uploaded_files ORDER BY datetime_uploaded DESC")
            db_files = c.fetchall()
            if not db_files:
                st.warning("Database kosong. Silakan unggah file terlebih dahulu.")
            else:
                file_options_db = {f"{rec[1]} (Kolom: {rec[3]}, ID: {rec[0]})": rec for rec in db_files}
                selected_file_key_db = st.selectbox("Pilih file dari database:", list(file_options_db.keys()))
                if selected_file_key_db:
                    selected_rec = file_options_db[selected_file_key_db]
                    db_id, filename_db, filepath_db, text_col_db = selected_rec
                    
                    with st.spinner(f"Memuat '{filename_db}'..."):
                        df_loaded = load_df_from_path(filepath_db, filename_db)
                    if df_loaded is not None:
                        df_to_process = df_loaded
                        text_col_to_process = text_col_db # Use the stored text column
                        filename_for_processing = filename_db
                        
                        # Update session state data to reflect this choice for subsequent steps
                        st.session_state.data['raw_df'] = df_to_process
                        st.session_state.data['raw_df_path'] = filepath_db
                        st.session_state.data['raw_df_filename'] = filename_for_processing
                        st.session_state.data['raw_df_text_col'] = text_col_to_process
                        st.session_state.data['processed_df'] = None # Clear any previous processing
                        st.session_state.data['classified_df'] = None

                        st.info(f"Data dari DB dipilih: '{filename_for_processing}' (Kolom teks: '{text_col_to_process}')")
                        display_df_preview(df_to_process, "Data yang akan diproses:")
                    else:
                        st.error(f"Gagal memuat file '{filename_db}' dari path '{filepath_db}'.")
        except sqlite3.Error as e:
            st.error(f"Database error: {e}")

    if df_to_process is not None and text_col_to_process:
        if text_col_to_process not in df_to_process.columns:
            st.error(f"Kolom teks '{text_col_to_process}' tidak ditemukan dalam data '{filename_for_processing}'. Harap periksa kembali pilihan file atau kolom di Langkah 1.")
        elif st.button("✨ Mulai Proses Cleaning & Preprocessing Teks ✨", key="start_preprocessing_btn"):
            with st.spinner(f"Melakukan preprocessing pada '{filename_for_processing}'..."):
                df_processed = df_to_process.copy()
                # Ensure text column is string and handle NaNs before preprocessing
                df_processed[text_col_to_process] = df_processed[text_col_to_process].astype(str).fillna('')
                df_processed['processed_text'] = preprocess_text_series_st(df_processed[text_col_to_process], stemmer)
            
            st.session_state.data['processed_df'] = df_processed
            st.success("Preprocessing teks selesai!")
            display_df_preview(df_processed[[text_col_to_process, 'processed_text']], "Data Setelah Preprocessing:", 5)
            st.info("Lanjutkan ke 'Pelatihan Model SVM'.")
    elif df_to_process is None and source_choice == "Gunakan data yang baru diunggah/dikonfirmasi":
        pass # Warning already shown
    elif df_to_process is None and source_choice == "Pilih dari database" and db_files: # only if db_files was populated
        st.info("Pilih file dari database untuk melanjutkan.")


elif choice == "3. Pelatihan Model SVM (Pseudo-Labeling)":
    st.header("🏋️ 3. Pelatihan Model SVM (Pseudo-Labeling)")

    if st.session_state.data['processed_df'] is None:
        st.warning("Data belum diproses. Silakan lakukan preprocessing di menu '2. Preprocessing Data'.")
        st.stop()
    if sentiment_analyzer_pipeline is None:
        st.error("Pipeline model pre-trained sentiment tidak tersedia. Pelatihan tidak dapat dilanjutkan.")
        st.stop()

    df_processed = st.session_state.data['processed_df']
    original_filename = st.session_state.data['raw_df_filename']
    original_text_col = st.session_state.data['raw_df_text_col']

    st.info(f"Data dari '{original_filename}' (kolom '{original_text_col}') telah diproses.")
    st.write("Kolom 'processed_text' akan digunakan untuk pelatihan (5 baris pertama):")
    st.dataframe(df_processed[['processed_text']].head())

    st.subheader("Parameter Pelatihan SVM")
    col1, col2, col3 = st.columns(3)
    with col1:
        train_subset_ratio = st.slider("Rasio Data Latih (dari data diproses):", 0.05, 1.0, 0.20, 0.05, key="svm_train_ratio", help="Persentase data yang sudah diproses untuk pseudo-labeling dan training SVM.")
    with col2:
        max_features_tfidf = st.number_input("Fitur Maksimum (TF-IDF):", 100, 10000, 3000, 100, key="svm_max_features")
    with col3:
        svm_kernel = st.selectbox("Kernel SVM:", ['linear', 'rbf', 'poly'], key="svm_kernel_select")
    svm_c = st.number_input("Parameter C (Regularization) SVM:", 0.01, 100.0, 1.0, 0.1, format="%.2f", key="svm_c_param")

    if st.button("🚀 Mulai Pelatihan SVM dengan Pseudo-Labels", key="start_svm_train_btn"):
        if 'processed_text' not in df_processed.columns or df_processed['processed_text'].isnull().all():
            st.error("Kolom 'processed_text' kosong atau tidak ada. Preprocessing mungkin gagal.")
            st.stop()
        
        min_samples_for_training = max(5, int(len(df_processed) * 0.01)) # At least 5 samples or 1%
        if len(df_processed) < min_samples_for_training :
             st.error(f"Data yang diproses terlalu sedikit ({len(df_processed)} baris). Membutuhkan setidaknya {min_samples_for_training} baris untuk pelatihan.")
             st.stop()

        with st.spinner(f"Memilih {train_subset_ratio*100:.0f}% data, pseudo-labeling, dan melatih SVM..."):
            n_total = len(df_processed)
            n_subset = max(min_samples_for_training, int(n_total * train_subset_ratio)) # Ensure at least min_samples
            df_train_subset = df_processed.sample(n=n_subset, random_state=42)
            st.write(f"Jumlah data untuk pseudo-labeling & training: {len(df_train_subset)} dari {n_total}.")

            texts_for_pseudo = df_train_subset['processed_text'].tolist()
            pseudo_labels = []
            labeling_errors = 0
            
            progress_bar_pseudo = st.progress(0, text="Pseudo-labeling teks...")
            for i, text in enumerate(texts_for_pseudo):
                current_prog = (i + 1) / len(texts_for_pseudo)
                progress_bar_pseudo.progress(current_prog, text=f"Pseudo-labeling teks... {i+1}/{len(texts_for_pseudo)}")
                if pd.isna(text) or not str(text).strip():
                    pseudo_labels.append("neutral") # Default for empty/NaN
                else:
                    try:
                        result = sentiment_analyzer_pipeline(str(text))
                        pseudo_labels.append(result[0]['label'].lower())
                    except Exception:
                        labeling_errors += 1
                        pseudo_labels.append("neutral") # Default on error
            progress_bar_pseudo.empty()

            if labeling_errors > 0:
                st.warning(f"{labeling_errors} teks gagal di-pseudo-label, diberi label 'neutral'.")

            df_train_subset = df_train_subset.copy() # Avoid SettingWithCopyWarning
            df_train_subset.loc[:, 'pseudo_sentiment'] = pseudo_labels
            
            st.write("Contoh data setelah pseudo-labeling:")
            st.dataframe(df_train_subset[[original_text_col, 'processed_text', 'pseudo_sentiment']].head())

            label_encoder = LabelEncoder()
            df_train_subset.loc[:, 'pseudo_sentiment_encoded'] = label_encoder.fit_transform(df_train_subset['pseudo_sentiment'])
            st.session_state.pseudo_label_encoder = label_encoder
            
            st.write("Mapping Label (Pseudo) ke Angka:")
            st.json(dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

            tfidf_vectorizer = TfidfVectorizer(max_features=max_features_tfidf, ngram_range=(1,2), min_df=2, sublinear_tf=True) # min_df adjusted
            X_train = tfidf_vectorizer.fit_transform(df_train_subset['processed_text'].fillna(''))
            y_train = df_train_subset['pseudo_sentiment_encoded']
            st.session_state.tfidf_vectorizer = tfidf_vectorizer

            svm_model = SVC(kernel=svm_kernel, C=svm_c, probability=True, random_state=42, class_weight='balanced')
            svm_model.fit(X_train, y_train)
            st.session_state.svm_model = svm_model
            st.success("Model SVM berhasil dilatih dengan pseudo-label!")

            y_pred_train = svm_model.predict(X_train)
            accuracy_train = accuracy_score(y_train, y_pred_train)
            
            unique_labels_enc = sorted(y_train.unique())
            unique_target_names = label_encoder.inverse_transform(unique_labels_enc)
            
            report_train_dict = classification_report(
                y_train, y_pred_train, 
                labels=unique_labels_enc, 
                target_names=unique_target_names,
                output_dict=True, zero_division=0
            )
            st.session_state.svm_training_details = {
                'accuracy_on_pseudo_train': accuracy_train,
                'report_on_pseudo_train': report_train_dict,
                'df_pseudo_labeled_sample': df_train_subset[[original_text_col, 'processed_text', 'pseudo_sentiment']].head()
            }
            st.subheader("Evaluasi Model SVM (pada Data Training Pseudo-Labeled):")
            st.metric("Akurasi pada Data Training (vs Pseudo-Labels)", f"{accuracy_train:.4f}")
            st.text("Laporan Klasifikasi Detail (vs Pseudo-Labels):")
            st.dataframe(pd.DataFrame(report_train_dict).transpose().style.format("{:.2f}"))
            st.info("Model SVM, Vectorizer, dan Encoder disimpan. Lanjutkan ke Klasifikasi.")


elif choice == "4. Klasifikasi Sentimen dengan SVM":
    st.header("📊 4. Klasifikasi Sentimen dengan SVM")

    if not all([st.session_state.svm_model, st.session_state.tfidf_vectorizer, st.session_state.pseudo_label_encoder]):
        st.warning("Model SVM/komponennya belum dilatih/tersedia. Latih di Langkah 3.")
        st.stop()
    if st.session_state.data['processed_df'] is None:
        st.warning("Data belum diproses. Lakukan preprocessing di Langkah 2.")
        st.stop()

    df_to_classify = st.session_state.data['processed_df']
    original_filename = st.session_state.data['raw_df_filename']
    original_text_col = st.session_state.data['raw_df_text_col']

    st.info(f"Akan dilakukan klasifikasi pada seluruh data '{original_filename}' (yang sudah diproses).")
    st.write(f"Total data untuk klasifikasi: {len(df_to_classify)} baris.")
    display_df_preview(df_to_classify[[original_text_col, 'processed_text']], "Contoh data untuk diklasifikasi:", 3)

    if st.button("🔎 Mulai Klasifikasi Sentimen dengan SVM", key="start_svm_classify_btn"):
        if 'processed_text' not in df_to_classify.columns or df_to_classify['processed_text'].isnull().all():
            st.error("Kolom 'processed_text' tidak ada/kosong. Preprocessing mungkin gagal.")
            st.stop()

        with st.spinner("Mengaplikasikan TF-IDF dan memprediksi sentimen..."):
            df_classified = df_to_classify.copy()
            texts_for_classification = df_classified['processed_text'].fillna('')
            X_features = st.session_state.tfidf_vectorizer.transform(texts_for_classification)
            
            svm_predictions_encoded = st.session_state.svm_model.predict(X_features)
            svm_predictions_text = st.session_state.pseudo_label_encoder.inverse_transform(svm_predictions_encoded)
            
            df_classified['sentiment_svm_prediction'] = svm_predictions_text
            st.session_state.data['classified_df'] = df_classified
        
        st.success("Klasifikasi sentimen dengan SVM selesai!")
        display_cols_preview = [original_text_col, 'processed_text', 'sentiment_svm_prediction']
        display_cols_preview = [col for col in display_cols_preview if col in df_classified.columns] # Ensure cols exist
        display_df_preview(df_classified[display_cols_preview], "Contoh Hasil Klasifikasi SVM:", 5)
        st.info("Hasil lengkap ada di menu '5. Hasil Analisis Sentimen'.")


elif choice == "5. Hasil Analisis Sentimen":
    st.header("📈 5. Hasil Analisis Sentimen (dari Klasifikasi SVM)")

    if st.session_state.data['classified_df'] is None:
        st.warning("Belum ada data yang diklasifikasi. Jalankan Langkah 4.")
        st.stop()

    df_results = st.session_state.data['classified_df']
    original_filename = st.session_state.data['raw_df_filename']
    original_text_col = st.session_state.data['raw_df_text_col']

    st.subheader(f"Hasil Analisis untuk File: '{original_filename}'")

    if 'sentiment_svm_prediction' in df_results.columns:
        cols_to_display = []
        if original_text_col in df_results.columns:
            cols_to_display.append(original_text_col)
        if 'processed_text' in df_results.columns:
            cols_to_display.append('processed_text')
        cols_to_display.append('sentiment_svm_prediction')
        
        # Remove duplicates just in case, though unlikely with this setup
        cols_to_display = sorted(list(set(cols_to_display)), key=cols_to_display.index) 
        # Final check if columns actually exist
        cols_to_display = [col for col in cols_to_display if col in df_results.columns]


        st.subheader("Tabel Hasil Klasifikasi SVM:")
        if cols_to_display:
            st.dataframe(df_results[cols_to_display], height=400)
        else:
            st.warning("Tidak dapat menampilkan kolom hasil, periksa alur data.")


        if cols_to_display: # Download button only if there's something to download
            @st.cache_data
            def convert_df_for_download(df_to_convert, columns_to_include):
                return df_to_convert[columns_to_include].to_csv(index=False).encode('utf-8')

            csv_download = convert_df_for_download(df_results, cols_to_display)
            st.download_button(
                label="📥 Unduh Hasil Klasifikasi (CSV)",
                data=csv_download,
                file_name=f"hasil_svm_{original_filename.split('.')[0]}.csv",
                mime='text/csv',
                key="download_svm_results_csv"
            )
        
        st.markdown("---")
        st.subheader("Distribusi Sentimen Hasil Klasifikasi SVM:")
        sentiment_counts = df_results['sentiment_svm_prediction'].value_counts()

        if not sentiment_counts.empty:
            # st.bar_chart(sentiment_counts) # Streamlit's basic bar chart

            # Enhanced plot with Seaborn
            try:
                fig, ax = plt.subplots(figsize=(8, 5)) # Adjusted size
                sns.countplot(
                    x='sentiment_svm_prediction',
                    data=df_results,
                    order=sentiment_counts.index,
                    ax=ax,
                    palette="pastel"
                )
                ax.set_title('Distribusi Sentimen (Prediksi SVM)', fontsize=14)
                ax.set_xlabel('Sentimen', fontsize=12)
                ax.set_ylabel('Jumlah Komentar', fontsize=12)
                for p in ax.patches:
                    ax.annotate(f'{int(p.get_height())}',
                                (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='center',
                                xytext=(0, 5),
                                textcoords='offset points', fontsize=10)
                plt.xticks(rotation=20, ha='right', fontsize=10)
                plt.yticks(fontsize=10)
                plt.tight_layout()
                st.pyplot(fig)
            except Exception as e_plot:
                st.error(f"Gagal membuat plot distribusi: {e_plot}")
                st.bar_chart(sentiment_counts) # Fallback to simple chart

        else:
            st.info("Tidak ada data sentimen valid untuk visualisasi.")
    else:
        st.error("Kolom 'sentiment_svm_prediction' tidak ditemukan. Klasifikasi mungkin belum berhasil.")

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.info("Aplikasi Analisis Sentimen v2.2-s\n(SVM dengan Pseudo-Labeling - Sederhana)")