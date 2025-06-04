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

# Sambungan dari kode yang Anda berikan sebelumnya...
# Pastikan semua import, inisialisasi global (stemmer, pipeline HF, koneksi DB),
# dan fungsi helper sudah ada di atas bagian ini.
# Sambungan dari kode yang Anda berikan sebelumnya...
# Pastikan semua import, inisialisasi global (stemmer, pipeline HF, koneksi DB),
# dan fungsi helper sudah ada di atas bagian ini.

elif choice == "2. Preprocessing Data":
    st.header("🧹 2. Preprocessing Data Teks")

    options_for_preprocessing = {"Pilih data untuk diproses...": None}
    # Untuk menyimpan detail setiap opsi agar mudah diakses setelah dipilih
    source_details_map = {}

    # Kandidat 1: Data yang baru diunggah (dari session state)
    if st.session_state.get('uploaded_file_info') and 'df' in st.session_state.uploaded_file_info:
        up_info = st.session_state.uploaded_file_info
        # Pastikan 'df' dan 'text_col' ada dan valid
        if isinstance(up_info.get('df'), pd.DataFrame) and up_info.get('text_col'):
            option_key_new = f"Data Baru Diunggah: {up_info['name']} (Kolom Teks: {up_info['text_col']})"
            options_for_preprocessing[option_key_new] = "newly_uploaded"
            source_details_map["newly_uploaded"] = {
                'df': up_info['df'],
                'text_col_suggestion': up_info['text_col'],
                'original_filename': up_info['name'],
                'source_type': 'newly_uploaded',
                'db_id': up_info.get('id') # Jika ada ID dari DB
            }
        else: # Jika info tidak lengkap, tawarkan untuk re-upload atau pilih dari DB
            st.warning("Informasi data yang baru diunggah tidak lengkap. Harap konfirmasi ulang di menu '1. Upload Data' atau pilih dari database.")
            # Reset agar tidak terjebak
            if st.button("Bersihkan Seleksi Data Baru Diunggah", key="clear_new_upload_for_pp"):
                st.session_state.uploaded_file_info = None
                st.rerun()


    # Kandidat 2...N: File dari database
    try:
        c.execute("SELECT id, filename, filepath, original_text_column FROM uploaded_files ORDER BY datetime_uploaded DESC")
        db_files = c.fetchall()
        for rec in db_files:
            db_id, filename_db, filepath_db, text_col_db_original = rec
            option_key_db = f"Dari DB: {filename_db} (ID: {db_id}) - Kolom Teks Asli: {text_col_db_original or 'Belum Ditentukan'}"
            options_for_preprocessing[option_key_db] = f"db_{db_id}" # Kunci unik untuk setiap file DB
            source_details_map[f"db_{db_id}"] = {
                'filepath': filepath_db, # Akan dimuat nanti jika dipilih
                'text_col_suggestion': text_col_db_original,
                'original_filename': filename_db,
                'source_type': 'database_file',
                'db_id': db_id,
                'filename_hint_for_loader': filename_db # Untuk helper load_df
            }
    except sqlite3.Error as e:
        st.error(f"Tidak dapat mengakses database untuk memuat daftar file: {e}")
        db_files = [] # Pastikan db_files terdefinisi

    if len(options_for_preprocessing) == 1 and "Pilih data untuk diproses..." in options_for_preprocessing : # Hanya opsi default
        st.warning("Tidak ada data yang tersedia untuk diproses. Silakan unggah dan konfirmasi file di menu '1. Upload Data'.")
    else:
        selected_option_display_key = st.selectbox(
            "Pilih sumber data untuk preprocessing:",
            list(options_for_preprocessing.keys()), # Tampilkan semua kunci display
            key="unified_preprocessing_source_select"
        )

        # Inisialisasi variabel untuk data yang akan diproses
        df_to_process_candidate = None
        text_col_suggestion_for_input = ""
        original_filename_for_processing = ""
        selected_source_internal_key = None # Kunci internal seperti "newly_uploaded" atau "db_ID"

        if selected_option_display_key != "Pilih data untuk diproses...":
            selected_source_internal_key = options_for_preprocessing[selected_option_display_key]
            details = source_details_map.get(selected_source_internal_key)

            if details:
                original_filename_for_processing = details['original_filename']
                text_col_suggestion_for_input = details['text_col_suggestion']

                if details['source_type'] == 'newly_uploaded':
                    df_to_process_candidate = details['df']
                elif details['source_type'] == 'database_file':
                    with st.spinner(f"Memuat data '{details['original_filename']}' dari database..."):
                        df_to_process_candidate = load_df_from_path(details['filepath'], details['filename_hint_for_loader'])
                
                if df_to_process_candidate is not None and not df_to_process_candidate.empty:
                    st.info(f"Data yang dipilih: '{original_filename_for_processing}' (Sumber: {details['source_type'].replace('_', ' ').capitalize()})")
                    st.dataframe(df_to_process_candidate.head(3))

                    # Meminta konfirmasi/input untuk kolom teks yang akan diproses
                    # Jika text_col_suggestion_for_input tidak ada di kolom df, set ke string kosong agar user mengisi
                    if text_col_suggestion_for_input not in df_to_process_candidate.columns:
                        st.warning(f"Kolom teks yang disarankan ('{text_col_suggestion_for_input}') tidak ditemukan di file. Harap pilih secara manual.")
                        text_col_suggestion_for_input = df_to_process_candidate.columns[0] if len(df_to_process_candidate.columns) > 0 else ""


                    # Izinkan pengguna memilih kolom teks jika autodetect salah atau tidak ada
                    available_cols = df_to_process_candidate.columns.tolist()
                    current_text_col_index = 0
                    if text_col_suggestion_for_input and text_col_suggestion_for_input in available_cols:
                        current_text_col_index = available_cols.index(text_col_suggestion_for_input)
                    
                    final_text_column_to_process = st.selectbox(
                        "Konfirmasi atau pilih kolom teks yang akan diproses:",
                        options=available_cols,
                        index=current_text_col_index,
                        key="final_text_col_for_pp_select"
                    )
                    
                    if st.button("✨ Mulai Proses Cleaning & Preprocessing Teks ✨", key="start_final_preprocessing_btn"):
                        if not final_text_column_to_process:
                            st.error("Kolom teks belum dipilih atau kosong. Harap pilih kolom teks yang valid.")
                        else:
                            st.session_state.selected_text_col_preprocessing = final_text_column_to_process # Simpan nama kolom aktual yang diproses
                            
                            with st.spinner(f"Melakukan preprocessing pada {len(df_to_process_candidate)} baris. Mohon tunggu..."):
                                df_processed_output = df_to_process_candidate.copy()
                                df_processed_output[final_text_column_to_process] = df_processed_output[final_text_column_to_process].astype(str).fillna('') # Pastikan string dan handle NaN
                                df_processed_output['processed_text'] = preprocess_text_series_st(df_processed_output[final_text_column_to_process], stemmer)

                            st.session_state.processed_df_info = {
                                'original_filename': original_filename_for_processing,
                                'df': df_processed_output,
                                'original_text_column_name': final_text_column_to_process # Simpan nama kolom teks asli yang DIPROSES
                            }
                            st.success("Preprocessing teks selesai!")
                            st.subheader("Data Setelah Preprocessing (5 baris pertama):")
                            st.dataframe(df_processed_output[[final_text_column_to_process, 'processed_text']].head())
                            st.info(f"Jumlah baris data yang diproses: {len(df_processed_output)}. Silakan lanjutkan ke 'Pelatihan Model SVM'.")
                            
                            # Setelah diproses, data "baru diunggah" dianggap sudah "digunakan"
                            if selected_source_internal_key == "newly_uploaded":
                                st.session_state.uploaded_file_info = None 
                                st.info("Data baru yang diunggah telah diproses. Untuk memprosesnya lagi, pilih dari daftar database (jika disimpan) atau unggah ulang.")
                elif df_to_process_candidate is None and details: # Jika gagal load_df_from_path
                     st.error(f"Gagal memuat data untuk '{original_filename_for_processing}'. Periksa pesan error di atas.")


elif choice == "3. Pelatihan Model SVM (Pseudo-Labeling)":
    st.header("🏋️ 3. Pelatihan Model SVM (Pseudo-Labeling)")

    if st.session_state.processed_df_info is None:
        st.warning("Data belum diproses. Silakan lakukan preprocessing di menu '2. Preprocessing Data'.")
        st.stop()
    if sentiment_analyzer_pipeline is None: 
        st.error("Pipeline model pre-trained sentiment tidak tersedia (gagal dimuat saat awal). Pelatihan tidak dapat dilanjutkan.")
        st.stop()

    processed_info_train = st.session_state.processed_df_info
    df_processed_full = processed_info_train['df']
    original_filename_train = processed_info_train['original_filename']
    # Nama kolom teks asli yang benar-benar diproses di tahap sebelumnya
    original_text_col_name_actual = processed_info_train.get('original_text_column_name', 'Teks Asli')


    st.info(f"Data yang digunakan untuk pelatihan berasal dari file '{original_filename_train}' (kolom 'processed_text').")
    st.write("Contoh data 'processed_text' yang akan digunakan (5 baris pertama):")
    st.dataframe(df_processed_full[['processed_text']].head())

    # Parameter Pelatihan
    st.subheader("Parameter Pelatihan SVM")
    col1_train, col2_train, col3_train = st.columns(3)
    with col1_train:
        train_subset_ratio = st.slider("Rasio Data Latih (dari data diproses):", 0.05, 1.0, 0.20, 0.05, key="train_subset_ratio_slider", help="Persentase data yang sudah diproses untuk dijadikan data latih setelah pseudo-labeling.")
    with col2_train:
        max_features_tfidf = st.number_input("Fitur Maksimum (TF-IDF):", min_value=100, max_value=10000, value=3000, step=100, key="max_feat_tfidf")
    with col3_train:
        svm_kernel_train = st.selectbox("Kernel SVM:", options=['linear', 'rbf', 'poly'], index=0, key="svm_kernel_train_select")
    
    svm_c_train = st.number_input("Parameter C (Regularization) SVM:", min_value=0.01, value=1.0, step=0.1, format="%.2f", key="svm_c_train_input", help="Nilai C yang lebih kecil menghasilkan margin yang lebih lebar (lebih banyak misklasifikasi pada data latih).")


    if st.button("🚀 Mulai Pelatihan SVM dengan Pseudo-Labels", key="start_svm_pseudo_train_btn"):
        if 'processed_text' not in df_processed_full.columns or df_processed_full['processed_text'].isnull().all():
            st.error("Kolom 'processed_text' tidak ada atau kosong. Pastikan preprocessing berhasil.")
            st.stop()
        
        min_data_for_training_svm = 10 
        if len(df_processed_full) < min_data_for_training_svm:
             st.error(f"Data yang diproses terlalu sedikit (kurang dari {min_data_for_training_svm} baris). Tidak cukup untuk pelatihan.")
             st.stop()

        with st.spinner(f"Memilih {train_subset_ratio*100:.0f}% data, melakukan pseudo-labeling, dan melatih SVM... Ini mungkin memakan waktu."):
            
            n_total_samples_train = len(df_processed_full)
            # Pastikan minimal 5 sampel, atau semua data jika n_total_samples_train < 5
            n_train_samples_actual = max(min(5, n_total_samples_train), int(n_total_samples_train * train_subset_ratio))
            if n_train_samples_actual == 0 and n_total_samples_train > 0 : n_train_samples_actual = 1 # Safety for very small df

            df_train_subset_svm = df_processed_full.sample(n=n_train_samples_actual, random_state=42)
            st.write(f"Jumlah data untuk pseudo-labeling & training SVM: {len(df_train_subset_svm)} dari {n_total_samples_train} total data yang diproses.")

            texts_for_pseudo_labeling_svm = df_train_subset_svm['processed_text'].tolist()
            pseudo_labels_list_svm = []
            labeling_errors_count_svm = 0

            progress_bar_pseudo_svm = st.progress(0, text="Pseudo-labeling teks...")
            for i, text_to_label_svm in enumerate(texts_for_pseudo_labeling_svm):
                current_progress_val = (i + 1) / len(texts_for_pseudo_labeling_svm)
                progress_bar_pseudo_svm.progress(current_progress_val, text=f"Pseudo-labeling teks... {i+1}/{len(texts_for_pseudo_labeling_svm)} ({current_progress_val:.0%})")
                if pd.isna(text_to_label_svm) or not str(text_to_label_svm).strip():
                    pseudo_labels_list_svm.append("neutral")
                else:
                    try:
                        result_svm = sentiment_analyzer_pipeline(str(text_to_label_svm))
                        pseudo_labels_list_svm.append(result_svm[0]['label'].lower())
                    except Exception:
                        labeling_errors_count_svm += 1
                        pseudo_labels_list_svm.append("neutral")
            progress_bar_pseudo_svm.empty()

            if labeling_errors_count_svm > 0:
                st.warning(f"{labeling_errors_count_svm} dari {len(texts_for_pseudo_labeling_svm)} teks gagal di-pseudo-label dan diberi label 'neutral'.")

            df_train_subset_svm = df_train_subset_svm.copy()
            df_train_subset_svm.loc[:, 'pseudo_sentiment'] = pseudo_labels_list_svm
            
            st.write("Contoh data setelah pseudo-labeling (5 baris pertama):")
            st.dataframe(df_train_subset_svm[[original_text_col_name_actual, 'processed_text', 'pseudo_sentiment']].head())

            current_pseudo_label_encoder_svm = LabelEncoder()
            df_train_subset_svm.loc[:, 'pseudo_sentiment_encoded'] = current_pseudo_label_encoder_svm.fit_transform(df_train_subset_svm['pseudo_sentiment'])
            st.session_state.pseudo_label_encoder = current_pseudo_label_encoder_svm
            
            st.write("Mapping Label (dari Pseudo-Labeling) ke Angka:")
            label_mapping_svm = dict(zip(st.session_state.pseudo_label_encoder.classes_, st.session_state.pseudo_label_encoder.transform(st.session_state.pseudo_label_encoder.classes_)))
            st.json(label_mapping_svm)

            current_tfidf_vectorizer_svm = TfidfVectorizer(max_features=max_features_tfidf, ngram_range=(1, 2), min_df=3, sublinear_tf=True) # Naikkan min_df
            X_train_for_svm_model = current_tfidf_vectorizer_svm.fit_transform(df_train_subset_svm['processed_text'].fillna(''))
            y_train_for_svm_model = df_train_subset_svm['pseudo_sentiment_encoded']
            st.session_state.tfidf_vectorizer = current_tfidf_vectorizer_svm

            current_svm_model_trained = SVC(kernel=svm_kernel_train, C=svm_c_train, probability=True, random_state=42, class_weight='balanced')
            current_svm_model_trained.fit(X_train_for_svm_model, y_train_for_svm_model)
            st.session_state.svm_model = current_svm_model_trained

            st.success("Model SVM berhasil dilatih dengan pseudo-label!")

            y_pred_on_pseudo_train_svm = st.session_state.svm_model.predict(X_train_for_svm_model)
            accuracy_on_pseudo_train_svm = accuracy_score(y_train_for_svm_model, y_pred_on_pseudo_train_svm)
            
            unique_labels_encoded_svm = sorted(y_train_for_svm_model.unique())
            unique_target_names_svm = st.session_state.pseudo_label_encoder.inverse_transform(unique_labels_encoded_svm)

            report_on_pseudo_train_dict_svm = classification_report(
                y_train_for_svm_model, 
                y_pred_on_pseudo_train_svm,
                labels=unique_labels_encoded_svm,
                target_names=unique_target_names_svm,
                output_dict=True, 
                zero_division=0
            )
            
            st.session_state.svm_training_data = {
                'df_train_pseudo_labeled': df_train_subset_svm.copy(),
                'original_text_column_used': original_text_col_name_actual,
                'accuracy_on_pseudo_train_set': accuracy_on_pseudo_train_svm,
                'report_on_pseudo_train_set': report_on_pseudo_train_dict_svm
            }
            st.subheader("Evaluasi Kinerja Model SVM (pada Data Training yang di-Pseudo-Label):")
            st.metric("Akurasi pada Data Training (vs Pseudo-Labels)", f"{accuracy_on_pseudo_train_svm:.4f}")
            st.text("Laporan Klasifikasi Detail pada Data Training (vs Pseudo-Labels):")
            st.dataframe(pd.DataFrame(report_on_pseudo_train_dict_svm).transpose().style.format("{:.2f}"))
            st.info("Model SVM, Vectorizer TF-IDF, dan Label Encoder telah berhasil disimpan. Anda dapat melanjutkan ke menu 'Klasifikasi Sentimen'.")


elif choice == "4. Klasifikasi Sentimen dengan SVM":
    st.header("📊 4. Klasifikasi Sentimen Menggunakan Model SVM yang Telah Dilatih")

    if not st.session_state.get('svm_model') or \
       not st.session_state.get('tfidf_vectorizer') or \
       not st.session_state.get('pseudo_label_encoder'):
        st.warning("Komponen Model SVM (model, vectorizer, atau encoder) belum tersedia. Silakan latih model terlebih dahulu di menu '3. Pelatihan Model SVM'.")
        st.stop()
    
    if not st.session_state.get('processed_df_info'):
        st.warning("Data belum diproses. Silakan unggah dan proses data terlebih dahulu melalui menu '1. Upload Data' dan '2. Preprocessing Data'.")
        st.stop()

    data_to_classify_info_svm = st.session_state.processed_df_info
    df_processed_for_classification_svm = data_to_classify_info_svm['df']
    original_filename_for_classification_svm = data_to_classify_info_svm['original_filename']
    original_text_col_name_for_display_classify_svm = data_to_classify_info_svm.get('original_text_column_name', 'Teks Asli Tidak Diketahui')

    st.info(f"Akan dilakukan klasifikasi sentimen pada seluruh data dari file '{original_filename_for_classification_svm}' (yang telah melalui preprocessing) menggunakan model SVM yang telah dilatih.")
    st.write(f"Jumlah total data yang akan diklasifikasi: {len(df_processed_for_classification_svm)} baris.")
    st.write("Contoh data 'processed_text' yang akan diklasifikasi (5 baris pertama):")
    st.dataframe(df_processed_for_classification_svm[[original_text_col_name_for_display_classify_svm,'processed_text']].head())

    if st.button("🔎 Mulai Klasifikasi Sentimen dengan SVM pada Seluruh Data", key="start_svm_full_data_classification_btn"):
        if 'processed_text' not in df_processed_for_classification_svm.columns or df_processed_for_classification_svm['processed_text'].isnull().all():
            st.error("Kolom 'processed_text' tidak ditemukan atau seluruhnya kosong pada data yang akan diklasifikasi. Pastikan preprocessing berhasil.")
            st.stop()

        with st.spinner("Mengaplikasikan TF-IDF dan memprediksi sentimen dengan SVM pada seluruh data... Ini mungkin memakan waktu."):
            df_classified_output_svm = df_processed_for_classification_svm.copy()

            texts_for_svm_classification_full = df_classified_output_svm['processed_text'].fillna('')
            X_features_for_classification_full = st.session_state.tfidf_vectorizer.transform(texts_for_svm_classification_full)

            svm_predictions_encoded_output_full = st.session_state.svm_model.predict(X_features_for_classification_full)
            svm_predictions_text_output_full = st.session_state.pseudo_label_encoder.inverse_transform(svm_predictions_encoded_output_full)

            df_classified_output_svm['sentiment_svm_prediction'] = svm_predictions_text_output_full
            
            st.session_state.classified_df_info = {
                'original_filename': original_filename_for_classification_svm,
                'df': df_classified_output_svm,
                'original_text_column_name': original_text_col_name_for_display_classify_svm 
            }
        
        st.success("Klasifikasi sentimen dengan SVM pada seluruh data telah selesai!")
        st.subheader("Contoh Hasil Klasifikasi SVM (5 baris pertama):")
        
        display_cols_for_classify_preview_svm = [original_text_col_name_for_display_classify_svm, 'processed_text', 'sentiment_svm_prediction']
        display_cols_for_classify_preview_svm = [col for col in display_cols_for_classify_preview_svm if col in df_classified_output_svm.columns]
        
        st.dataframe(df_classified_output_svm[display_cols_for_classify_preview_svm].head())
        st.info("Hasil klasifikasi lengkap dapat dilihat dan diunduh di menu '5. Hasil Analisis Sentimen'.")


elif choice == "5. Hasil Analisis Sentimen":
    st.header("📈 5. Hasil Analisis Sentimen (dari Klasifikasi SVM)")

    if not st.session_state.get('classified_df_info'):
        st.warning("Belum ada data yang diklasifikasi menggunakan SVM. Silakan jalankan proses klasifikasi di menu '4. Klasifikasi Sentimen dengan SVM' terlebih dahulu.")
        st.stop()

    final_results_info_display = st.session_state.classified_df_info
    df_final_svm_results_display = final_results_info_display['df']
    original_filename_final_results_display = final_results_info_display['original_filename']
    original_text_col_name_final_display_svm = final_results_info_display.get('original_text_column_name', 'Teks Asli Tidak Diketahui')

    st.subheader(f"Menampilkan Hasil Analisis Sentimen untuk File: '{original_filename_final_results_display}'")

    if 'sentiment_svm_prediction' in df_final_svm_results_display.columns:
        cols_to_display_final_table_svm = []
        # Coba ambil nama kolom asli dari info yang disimpan
        if original_text_col_name_final_display_svm != 'Teks Asli Tidak Diketahui' and original_text_col_name_final_display_svm in df_final_svm_results_display.columns:
            cols_to_display_final_table_svm.append(original_text_col_name_final_display_svm)
        elif st.session_state.get('selected_text_col_preprocessing') and st.session_state.selected_text_col_preprocessing in df_final_svm_results_display.columns:
            # Fallback ke nama kolom dari info preprocessing jika ada dan valid
            cols_to_display_final_table_svm.append(st.session_state.selected_text_col_preprocessing)
        
        if 'processed_text' in df_final_svm_results_display.columns:
             cols_to_display_final_table_svm.append('processed_text')
        cols_to_display_final_table_svm.append('sentiment_svm_prediction')
        
        cols_to_display_final_table_svm = list(dict.fromkeys(cols_to_display_final_table_svm)) # Hapus duplikat jika ada
        cols_to_display_final_table_svm = [col for col in cols_to_display_final_table_svm if col in df_final_svm_results_display.columns] # Filter final

        st.subheader("Tabel Lengkap Hasil Klasifikasi Sentimen SVM:")
        if cols_to_display_final_table_svm:
            st.dataframe(df_final_svm_results_display[cols_to_display_final_table_svm], height=400)
        else:
            st.warning("Tidak dapat menentukan kolom yang valid untuk ditampilkan. Periksa alur data.")

        if cols_to_display_final_table_svm: # Tombol unduh hanya jika ada kolom untuk diunduh
            @st.cache_data 
            def convert_df_for_download_svm(df_to_convert, columns_to_include):
                return df_to_convert[columns_to_include].to_csv(index=False).encode('utf-8')

            csv_for_download_svm = convert_df_for_download_svm(df_final_svm_results_display, cols_to_display_final_table_svm)
            st.download_button(
                label="📥 Unduh Semua Hasil Klasifikasi (CSV)",
                data=csv_for_download_svm,
                file_name=f"hasil_klasifikasi_svm_{original_filename_final_results_display.split('.')[0]}.csv",
                mime='text/csv',
                key="download_all_classified_svm_results_final_csv"
            )
        st.markdown("---")
        st.subheader("Distribusi Sentimen Hasil Klasifikasi SVM pada Seluruh Data:")
        sentiment_svm_counts_display = df_final_svm_results_display['sentiment_svm_prediction'].value_counts()

        if not sentiment_svm_counts_display.empty:
            st.bar_chart(sentiment_svm_counts_display)
            try:
                fig_dist_svm, ax_dist_svm = plt.subplots(figsize=(9, 6)) # Ukuran disesuaikan
                palette_svm = sns.color_palette("pastel", len(sentiment_svm_counts_display))
                sns.countplot(
                    x='sentiment_svm_prediction', 
                    data=df_final_svm_results_display, 
                    order=sentiment_svm_counts_display.index,
                    ax=ax_dist_svm, 
                    palette=palette_svm
                )
                ax_dist_svm.set_title('Distribusi Sentimen (Prediksi SVM pada Seluruh Data)', fontsize=14)
                ax_dist_svm.set_xlabel('Sentimen', fontsize=12)
                ax_dist_svm.set_ylabel('Jumlah Komentar', fontsize=12)
                for p_svm in ax_dist_svm.patches:
                    ax_dist_svm.annotate(f'{int(p_svm.get_height())}', 
                                     (p_svm.get_x() + p_svm.get_width() / 2., p_svm.get_height()),
                                     ha='center', va='center', 
                                     xytext=(0, 5), 
                                     textcoords='offset points', fontsize=10, color='black')
                plt.xticks(rotation=30, ha='right', fontsize=10)
                plt.yticks(fontsize=10)
                plt.tight_layout()
                st.pyplot(fig_dist_svm)
            except Exception as e_plot_svm:
                st.error(f"Gagal membuat plot distribusi sentimen detail: {e_plot_svm}")
        else:
            st.info("Tidak ada data sentimen yang valid untuk divisualisasikan dari hasil klasifikasi SVM.")
    else:
        st.error("Kolom 'sentiment_svm_prediction' tidak ditemukan dalam data hasil. Proses klasifikasi mungkin belum berhasil atau tidak lengkap.")

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.info("Aplikasi Analisis Sentimen v2.2\n\n(SVM dengan Pseudo-Labeling)")