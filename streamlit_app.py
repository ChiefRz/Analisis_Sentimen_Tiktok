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
def load_hf_sentiment_pipeline():
    try:
        sentiment_pipeline = hf_pipeline(
            "sentiment-analysis",
            model="w11wo/indonesian-roberta-base-sentiment-classifier",
            tokenizer="w11wo/indonesian-roberta-base-sentiment-classifier"
        )
        return sentiment_pipeline
    except Exception as e:
        st.sidebar.error(f"Gagal memuat model pre-trained: {e}. Beberapa fitur mungkin tidak berfungsi.")
        return None

sentiment_analyzer_pipeline = load_hf_sentiment_pipeline()
if sentiment_analyzer_pipeline:
    st.sidebar.success("Model pre-trained (sentiment) siap.")

# --- Session State Initialization ---
# More granular session state variables
if 'uploaded_file_info' not in st.session_state:
    st.session_state.uploaded_file_info = None # {'id': db_id, 'name': str, 'path': str, 'df': pd.DataFrame}
if 'selected_text_col_preprocessing' not in st.session_state:
    st.session_state.selected_text_col_preprocessing = None
if 'processed_df_info' not in st.session_state: # For df with 'processed_text'
    st.session_state.processed_df_info = None # {'original_filename': str, 'df': pd.DataFrame}

# For SVM training
if 'svm_training_data' not in st.session_state: # The 20% pseudo-labeled data
    st.session_state.svm_training_data = None # {'df_train_pseudo': pd.DataFrame, 'report': dict, 'accuracy': float}
if 'tfidf_vectorizer' not in st.session_state:
    st.session_state.tfidf_vectorizer = None
if 'svm_model' not in st.session_state:
    st.session_state.svm_model = None
if 'pseudo_label_encoder' not in st.session_state: # LabelEncoder for pseudo_labels
    st.session_state.pseudo_label_encoder = None

# For SVM classification
if 'classification_target_df_info' not in st.session_state: # DF to be classified
     st.session_state.classification_target_df_info = None # {'original_filename': str, 'df': pd.DataFrame}
if 'classified_df_info' not in st.session_state: # DF with SVM predictions
    st.session_state.classified_df_info = None # {'original_filename': str, 'df': pd.DataFrame}


# --- Helper Functions ---
@st.cache_data(ttl=3600)
def load_df_from_path(file_path, filename_hint=""):
    try:
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path, on_bad_lines='skip')
        elif file_path.endswith('.xlsx'):
            return pd.read_excel(file_path)
        elif file_path.endswith('.txt'):
            # Try to infer delimiter, or assume one line per entry if single column
            try:
                return pd.read_csv(file_path, sep="\t", on_bad_lines='skip', header=None, names=['text'])
            except pd.errors.ParserError:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = [line.strip() for line in f if line.strip()]
                return pd.DataFrame(lines, columns=['text'])
        st.error(f"Format file tidak didukung: {file_path}")
        return None
    except pd.errors.EmptyDataError:
        st.error(f"File '{filename_hint}' kosong.")
        return None
    except Exception as e:
        st.error(f"Error saat membaca file '{filename_hint}': {e}")
        return None

@st.cache_data(ttl=3600) # Cache the preprocessing result for a given text
def _preprocess_text_cached(text, _sastrawi_stemmer_obj): # Pass stemmer for cache safety
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+', '', text)
    # text = re.sub(r'\#\w+', '', text) # Keep hashtags for now, could be sentiment indicators
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    text = text.translate(str.maketrans('', '', string.punctuation.replace("#", ""))) # Keep #
    text = re.sub(r'\d+', '', text) # Remove numbers
    text = re.sub(r'\s+', ' ', text).strip() # Normalize whitespace
    tokens = word_tokenize(text)
    stop_words_indonesian = set(stopwords.words('indonesian'))
    # Stemming after stopword removal is common
    # filtered_tokens = [word for word in tokens if word not in stop_words_indonesian and len(word) > 2]
    # stemmed_tokens = [_sastrawi_stemmer_obj.stem(token) for token in filtered_tokens]
    # Or stem first, then remove stopwords (can sometimes be better)
    stemmed_tokens = [_sastrawi_stemmer_obj.stem(token) for token in tokens]
    filtered_stemmed_tokens = [word for word in stemmed_tokens if word not in stop_words_indonesian and len(word) > 1]
    return ' '.join(filtered_stemmed_tokens)

def preprocess_text_series_st(text_series, sastrawi_stemmer_obj):
    processed_texts = []
    total = len(text_series)
    if total == 0: return []
    progress_bar = st.progress(0, text="Memproses teks...")
    for i, text_content in enumerate(text_series):
        processed_texts.append(_preprocess_text_cached(text_content, sastrawi_stemmer_obj))
        progress_bar.progress((i + 1) / total, text=f"Memproses teks... {i+1}/{total}")
    progress_bar.empty()
    return processed_texts

# --- Judul Aplikasi ---
st.set_page_config(page_title="Analisis Sentimen SVM", layout="wide")
st.title("📱 Aplikasi Analisis Sentimen (SVM & Pseudo-Labeling)")

# --- Pilihan Menu ---
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
    uploaded_file_obj = st.file_uploader("Pilih file (CSV, XLSX, TXT):", type=['csv', 'xlsx', 'txt'])

    if uploaded_file_obj is not None:
        filename = uploaded_file_obj.name
        file_path = os.path.join(UPLOAD_DIR, filename)

        with open(file_path, "wb") as f:
            f.write(uploaded_file_obj.getbuffer())

        df = load_df_from_path(file_path, filename)

        if df is not None and not df.empty:
            st.success(f"File '{filename}' berhasil diunggah dan dibaca.")
            st.subheader("Pratinjau Data:")
            st.dataframe(df.head())
            st.info(f"Jumlah baris: {len(df)}, Jumlah kolom: {len(df.columns)}")

            # Auto-detect or let user select text column
            common_text_cols = ['text', 'teks', 'review', 'tweet', 'komentar', 'ulasan', 'content', 'message']
            detected_text_col = None
            for col in common_text_cols:
                if col in df.columns.str.lower().tolist(): # case-insensitive check
                    # find the actual column name if different case
                    actual_col_name = [c for c in df.columns if c.lower() == col][0]
                    detected_text_col = actual_col_name
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
                        # Check if file already exists in DB to avoid re-inserting same physical file
                        # with different text col choice, just update if needed or handle appropriately.
                        # For simplicity, we allow re-uploading and new DB entry if user confirms.
                        c.execute("INSERT INTO uploaded_files (filename, filepath, original_text_column) VALUES (?, ?, ?)",
                                  (filename, file_path, selected_col))
                        conn.commit()
                        db_id = c.lastrowid
                        st.session_state.uploaded_file_info = {'id': db_id, 'name': filename, 'path': file_path, 'df': df, 'text_col': selected_col}
                        st.success(f"File '{filename}' dengan kolom teks '{selected_col}' dikonfirmasi dan disimpan ke database (ID: {db_id}). Lanjutkan ke Preprocessing.")
                    except sqlite3.IntegrityError:
                        st.warning(f"File dengan nama '{filename}' sudah ada di database. Jika ini file yang berbeda, ubah nama file sebelum mengunggah. Jika ini file yang sama, Anda bisa lanjut ke Preprocessing jika sudah dikonfirmasi sebelumnya.")
                        # Try to load existing info for this filename if integrity error
                        c.execute("SELECT id, filepath, original_text_column FROM uploaded_files WHERE filename = ?", (filename,))
                        existing_rec = c.fetchone()
                        if existing_rec:
                            st.session_state.uploaded_file_info = {'id': existing_rec[0], 'name': filename, 'path': existing_rec[1], 'df': df, 'text_col': existing_rec[2]}
                            st.info(f"Menggunakan data file '{filename}' yang sudah ada di DB dengan kolom teks '{existing_rec[2]}'.")

                else:
                    st.error("Harap pilih kolom teks yang valid.")
        elif df is None:
            st.error("Gagal memuat data dari file. File mungkin rusak atau format tidak didukung.")

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

    # Option to load from session state if upload just happened
    data_to_load_for_preprocessing = None
    source_name = ""

    if st.session_state.get('uploaded_file_info') and 'df' in st.session_state.uploaded_file_info:
        st.info(f"Menggunakan data yang baru diunggah: '{st.session_state.uploaded_file_info['name']}' dengan kolom teks '{st.session_state.uploaded_file_info['text_col']}'.")
        if st.button("Proses Data Ini", key="process_current_upload"):
            data_to_load_for_preprocessing = st.session_state.uploaded_file_info['df']
            st.session_state.selected_text_col_preprocessing = st.session_state.uploaded_file_info['text_col']
            source_name = st.session_state.uploaded_file_info['name']

    st.markdown("---")
    st.subheader("Atau pilih file dari database untuk diproses ulang:")
    c.execute("SELECT id, filename, filepath, original_text_column FROM uploaded_files ORDER BY id DESC")
    db_files = c.fetchall()

    if not db_files and data_to_load_for_preprocessing is None:
        st.warning("Tidak ada file di database. Silakan unggah data terlebih dahulu.")
    else:
        file_options = {f"{rec[1]} (ID: {rec[0]}) - Kolom Teks: {rec[3]}": rec for rec in db_files}
        selected_db_file_key = st.selectbox(
            "Pilih file dari database:",
            ["Pilih..."] + list(file_options.keys()),
            key="preprocess_db_select"
        )

        if selected_db_file_key != "Pilih...":
            selected_record = file_options[selected_db_file_key]
            db_id, filename_db, filepath_db, text_col_db = selected_record
            if st.button(f"Proses File dari DB: {filename_db}", key=f"process_db_{db_id}"):
                df_from_db = load_df_from_path(filepath_db, filename_db)
                if df_from_db is not None and not df_from_db.empty:
                    data_to_load_for_preprocessing = df_from_db
                    st.session_state.selected_text_col_preprocessing = text_col_db
                    source_name = filename_db
                else:
                    st.error(f"Gagal memuat file '{filename_db}' dari database.")

    if data_to_load_for_preprocessing is not None and st.session_state.selected_text_col_preprocessing:
        df_original = data_to_load_for_preprocessing
        text_column = st.session_state.selected_text_col_preprocessing

        if text_column not in df_original.columns:
            st.error(f"Kolom '{text_column}' tidak ditemukan dalam data '{source_name}'. Kolom tersedia: {', '.join(df_original.columns.tolist())}")
        else:
            st.write(f"Akan memproses kolom **'{text_column}'** dari file **'{source_name}'**.")
            st.write("Contoh data sebelum preprocessing:")
            st.dataframe(df_original[[text_column]].head())

            with st.spinner(f"Melakukan preprocessing pada {len(df_original)} baris..."):
                df_processed = df_original.copy()
                df_processed['processed_text'] = preprocess_text_series_st(df_processed[text_column].astype(str), stemmer)

            st.session_state.processed_df_info = {'original_filename': source_name, 'df': df_processed}
            st.success("Preprocessing selesai!")
            st.subheader("Data Setelah Preprocessing:")
            st.dataframe(df_processed[[text_column, 'processed_text']].head())
            st.info(f"Jumlah baris: {len(df_processed)}. Silakan lanjutkan ke 'Pelatihan Model SVM'.")
            # Clear uploaded_file_info so it doesn't auto-select next time unless re-confirmed
            st.session_state.uploaded_file_info = None
    elif not data_to_load_for_preprocessing and (st.session_state.uploaded_file_info or db_files):
         st.info("Pilih data yang akan diproses dengan menekan tombol 'Proses Data Ini' atau 'Proses File dari DB'.")


elif choice == "3. Pelatihan Model SVM (Pseudo-Labeling)":
    st.header("🏋️ 3. Pelatihan Model SVM (Pseudo-Labeling)")

    if st.session_state.processed_df_info is None:
        st.warning("Data belum diproses. Silakan lakukan preprocessing di menu '2. Preprocessing Data'.")
        st.stop()
    if sentiment_analyzer_pipeline is None:
        st.error("Pipeline model pre-trained sentiment tidak tersedia. Tidak dapat melanjutkan.")
        st.stop()

    df_processed_full = st.session_state.processed_df_info['df']
    original_filename_train = st.session_state.processed_df_info['original_filename']
    text_column_original_name = st.session_state.selected_text_col_preprocessing # Assuming this holds the original name

    st.info(f"Data yang digunakan berasal dari file '{original_filename_train}' yang sudah diproses.")
    st.write("Contoh data 'processed_text' yang akan digunakan:")
    st.dataframe(df_processed_full[['processed_text']].head())

    test_size_pseudo = st.slider("Persentase data untuk di-pseudo-label dan training SVM:", 0.05, 0.50, 0.20, 0.05, key="pseudo_train_size")

    if st.button("🚀 Mulai Pelatihan SVM dengan Pseudo-Labels", key="start_svm_pseudo_train"):
        if 'processed_text' not in df_processed_full.columns or df_processed_full['processed_text'].isnull().all():
            st.error("Kolom 'processed_text' tidak ada atau kosong. Pastikan preprocessing berhasil.")
            st.stop()
        if len(df_processed_full) < 10:
             st.error("Data yang diproses terlalu sedikit (kurang dari 10 baris).")
             st.stop()

        with st.spinner(f"Memilih {test_size_pseudo*100:.0f}% data, melakukan pseudo-labeling, dan melatih SVM..."):
            # 1. Ambil subset untuk training
            # If df is small, use more data for training, up to n=5 minimum if possible
            n_samples = len(df_processed_full)
            train_n = max(5, int(n_samples * test_size_pseudo)) # At least 5 samples for training if possible
            if train_n >= n_samples and n_samples > 1 : train_n = n_samples -1 # ensure test set is not empty if stratify is used
            if train_n == 0 and n_samples > 0 : train_n = 1
            if n_samples <=1 :
                st.error("Data sangat sedikit untuk dibagi.")
                st.stop()


            # For pseudo-labeling, we take a sample. The rest can be used for testing later.
            # Here, we are training SVM on pseudo-labeled data.
            df_train_subset = df_processed_full.sample(n=train_n, random_state=42)
            st.write(f"Jumlah data untuk pseudo-labeling & training SVM: {len(df_train_subset)}")

            # 2. Pseudo-labeling
            texts_for_pseudo_labeling = df_train_subset['processed_text'].tolist()
            pseudo_labels_list = []
            labeling_errors = 0

            progress_bar_label = st.progress(0, text="Pseudo-labeling...")
            for i, text in enumerate(texts_for_pseudo_labeling):
                if pd.isna(text) or not str(text).strip():
                    pseudo_labels_list.append("neutral") # Default
                else:
                    try:
                        result = sentiment_analyzer_pipeline(str(text)) # Ensure text is string
                        pseudo_labels_list.append(result[0]['label'].lower()) # e.g. POSITIVE -> positive
                    except Exception:
                        labeling_errors += 1
                        pseudo_labels_list.append("neutral") # Default on error
                progress_bar_label.progress((i + 1) / len(texts_for_pseudo_labeling), text=f"Pseudo-labeling... {i+1}/{len(texts_for_pseudo_labeling)}")
            progress_bar_label.empty()
            if labeling_errors > 0:
                st.warning(f"{labeling_errors} teks gagal di-pseudo-label dan diberi label 'neutral'.")

            df_train_subset.loc[:, 'pseudo_sentiment'] = pseudo_labels_list # Use .loc for safe assignment
            st.write("Contoh data setelah pseudo-labeling:")
            st.dataframe(df_train_subset[['processed_text', 'pseudo_sentiment']].head())

            # 3. Encode Pseudo-Labels
            st.session_state.pseudo_label_encoder = LabelEncoder()
            df_train_subset.loc[:, 'pseudo_sentiment_encoded'] = st.session_state.pseudo_label_encoder.fit_transform(df_train_subset['pseudo_sentiment'])
            st.write("Mapping Label Asli ke Angka:")
            st.json(dict(zip(st.session_state.pseudo_label_encoder.classes_, st.session_state.pseudo_label_encoder.transform(st.session_state.pseudo_label_encoder.classes_))))


            # 4. TF-IDF Feature Extraction (FIT on this training subset only)
            st.session_state.tfidf_vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=2) # min_df to avoid rare words
            X_train_svm = st.session_state.tfidf_vectorizer.fit_transform(df_train_subset['processed_text'])
            y_train_svm = df_train_subset['pseudo_sentiment_encoded']

            # 5. Train SVM Model
            st.session_state.svm_model = SVC(kernel='linear', C=1.0, probability=True, random_state=42, class_weight='balanced')
            st.session_state.svm_model.fit(X_train_svm, y_train_svm)
            st.success("Model SVM berhasil dilatih dengan pseudo-label!")

            # (Optional) Evaluate SVM on its own training data (pseudo-labeled)
            y_pred_on_train = st.session_state.svm_model.predict(X_train_svm)
            accuracy_on_train = accuracy_score(y_train_svm, y_pred_on_train)
            report_on_train_dict = classification_report(y_train_svm, y_pred_on_train,
                                                        labels=st.session_state.pseudo_label_encoder.transform(st.session_state.pseudo_label_encoder.classes_),
                                                        target_names=st.session_state.pseudo_label_encoder.classes_,
                                                        output_dict=True, zero_division=0)
            st.session_state.svm_training_data = {
                'df_train_pseudo': df_train_subset, # Includes original text, processed_text, pseudo_sentiment, pseudo_sentiment_encoded
                'original_text_column': text_column_original_name,
                'accuracy_on_pseudo_train': accuracy_on_train,
                'report_on_pseudo_train': report_on_train_dict
            }
            st.subheader("Evaluasi Model SVM pada Data Training (yang di-pseudo-label):")
            st.metric("Akurasi pada Data Training (vs Pseudo-Labels)", f"{accuracy_on_train:.4f}")
            st.text("Laporan Klasifikasi pada Data Training (vs Pseudo-Labels):")
            st.dataframe(pd.DataFrame(report_on_train_dict).transpose().style.format("{:.2f}"))
            st.info("Model SVM, Vectorizer, dan Label Encoder telah disimpan. Lanjutkan ke 'Klasifikasi Sentimen'.")

elif choice == "4. Klasifikasi Sentimen dengan SVM":
    st.header("📊 4. Klasifikasi Sentimen dengan Model SVM Terlatih")

    if not st.session_state.get('svm_model') or \
       not st.session_state.get('tfidf_vectorizer') or \
       not st.session_state.get('pseudo_label_encoder'):
        st.warning("Model SVM belum dilatih. Silakan latih model di menu '3. Pelatihan Model SVM'.")
        st.stop()
    if not st.session_state.get('processed_df_info'):
        st.warning("Data belum diproses. Silakan proses data di menu '2. Preprocessing Data'.")
        st.stop()

    df_processed_full_classify = st.session_state.processed_df_info['df']
    original_filename_classify = st.session_state.processed_df_info['original_filename']
    original_text_column_name = st.session_state.selected_text_col_preprocessing # from preprocessing step

    st.info(f"Akan mengklasifikasikan seluruh data dari file '{original_filename_classify}' (yang sudah diproses) menggunakan model SVM yang telah dilatih.")
    st.write("Contoh data 'processed_text' yang akan diklasifikasi:")
    st.dataframe(df_processed_full_classify[['processed_text']].head())

    if st.button("🔎 Mulai Klasifikasi dengan SVM", key="start_svm_classification"):
        if 'processed_text' not in df_processed_full_classify.columns or df_processed_full_classify['processed_text'].isnull().all():
            st.error("Kolom 'processed_text' tidak ada atau kosong pada data yang akan diklasifikasi.")
            st.stop()

        with st.spinner("Mengaplikasikan TF-IDF dan memprediksi sentimen dengan SVM..."):
            df_to_classify_svm = df_processed_full_classify.copy() # Make a copy

            # 1. TF-IDF Transform (using THE FITTED vectorizer)
            texts_for_classification = df_to_classify_svm['processed_text'].fillna('') # Handle NaN just in case
            X_classify = st.session_state.tfidf_vectorizer.transform(texts_for_classification)

            # 2. SVM Predict
            svm_predictions_encoded = st.session_state.svm_model.predict(X_classify)

            # 3. Decode Predictions
            svm_predictions_text = st.session_state.pseudo_label_encoder.inverse_transform(svm_predictions_encoded)

            df_to_classify_svm['sentiment_svm_prediction'] = svm_predictions_text
            st.session_state.classified_df_info = {
                'original_filename': original_filename_classify,
                'df': df_to_classify_svm,
                'original_text_column_name': original_text_column_name # Save for display
            }
        st.success("Klasifikasi sentimen dengan SVM selesai!")
        st.subheader("Contoh Hasil Klasifikasi SVM:")
        display_cols_classify = [original_text_column_name, 'processed_text', 'sentiment_svm_prediction']
        # Ensure original text column is present from the copy
        if original_text_column_name not in df_to_classify_svm.columns:
             st.warning(f"Kolom teks asli '{original_text_column_name}' tidak ditemukan di data. Menampilkan tanpa kolom tersebut.")
             display_cols_classify = ['processed_text', 'sentiment_svm_prediction']

        st.dataframe(df_to_classify_svm[display_cols_classify].head())
        st.info("Hasil lengkap dapat dilihat dan diunduh di menu '5. Hasil Analisis Sentimen'.")


elif choice == "5. Hasil Analisis Sentimen":
    st.header("📈 5. Hasil Analisis Sentimen (Klasifikasi SVM)")

    if not st.session_state.get('classified_df_info'):
        st.warning("Belum ada data yang diklasifikasi. Silakan jalankan 'Klasifikasi Sentimen dengan SVM' terlebih dahulu.")
        st.stop()

    classified_info = st.session_state.classified_df_info
    df_final_results = classified_info['df']
    original_filename_results = classified_info['original_filename']
    original_text_col_results = classified_info.get('original_text_column_name', 'Teks Asli Tidak Diketahui')


    st.subheader(f"Hasil Klasifikasi SVM untuk File: '{original_filename_results}'")

    if 'sentiment_svm_prediction' in df_final_results.columns:
        # Display original text (if available), processed_text, and svm_prediction
        cols_to_display_final = []
        if original_text_col_results in df_final_results.columns:
            cols_to_display_final.append(original_text_col_results)
        else: # If original text col name was not passed or doesn't exist, try common names if they exist
            common_cols_try = [st.session_state.get('selected_text_col_preprocessing', 'text'), 'text', 'teks', 'komentar']
            for cct in common_cols_try:
                if cct in df_final_results.columns:
                    cols_to_display_final.append(cct)
                    original_text_col_results = cct # Update for download filename consistency
                    break
        
        cols_to_display_final.extend(['processed_text', 'sentiment_svm_prediction'])
        # Filter out columns that might not exist in the final df (e.g. if original text col was dropped)
        cols_to_display_final = [col for col in cols_to_display_final if col in df_final_results.columns]


        st.subheader("Tabel Hasil Klasifikasi:")
        st.dataframe(df_final_results[cols_to_display_final])

        # Download button
        @st.cache_data # Cache data for download
        def convert_df_to_csv(df_to_convert):
            return df_to_convert.to_csv(index=False).encode('utf-8')

        csv_download = convert_df_to_csv(df_final_results[cols_to_display_final])
        st.download_button(
            label="📥 Unduh Hasil Klasifikasi (CSV)",
            data=csv_download,
            file_name=f"analisis_sentimen_svm_{original_filename_results.split('.')[0]}.csv",
            mime='text/csv',
            key="download_classified_csv"
        )
        st.markdown("---")
        st.subheader("Distribusi Sentimen Hasil Klasifikasi SVM:")
        sentiment_counts = df_final_results['sentiment_svm_prediction'].value_counts()

        if not sentiment_counts.empty:
            st.bar_chart(sentiment_counts)

            try:
                fig, ax = plt.subplots(figsize=(8, 5))
                palette = sns.color_palette("pastel", len(sentiment_counts))
                sns.countplot(x='sentiment_svm_prediction', data=df_final_results, order=sentiment_counts.index, ax=ax, palette=palette)
                ax.set_title('Distribusi Sentimen (Prediksi SVM)')
                ax.set_xlabel('Sentimen')
                ax.set_ylabel('Jumlah Komentar')
                for p in ax.patches:
                    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=10)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Gagal membuat plot detail: {e}")
        else:
            st.info("Tidak ada data sentimen untuk divisualisasikan.")
    else:
        st.error("Kolom 'sentiment_svm_prediction' tidak ditemukan dalam data hasil. Proses klasifikasi mungkin belum lengkap.")

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.info("Aplikasi Analisis Sentimen v2.0\n\n(SVM dengan Pseudo-Labeling)")