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

elif choice == "2. Preprocessing Data":
    st.header("🧹 2. Preprocessing Data Teks")

    # Option to load from session state if upload just happened
    data_to_load_for_preprocessing = None
    source_name = ""
    current_text_col_for_pp = None # Akan diisi dari session state atau pilihan DB

    # Prioritaskan data yang baru saja diunggah dan dikonfirmasi
    if st.session_state.get('uploaded_file_info') and 'df' in st.session_state.uploaded_file_info:
        up_info = st.session_state.uploaded_file_info
        st.info(f"Data yang baru diunggah: '{up_info['name']}' dengan kolom teks '{up_info['text_col']}'.")
        if st.button("Gunakan Data Ini untuk Preprocessing", key="process_current_upload"):
            data_to_load_for_preprocessing = up_info['df']
            current_text_col_for_pp = up_info['text_col']
            source_name = up_info['name']
            # Simpan nama kolom teks yang akan diproses dari sumber ini
            st.session_state.selected_text_col_preprocessing = current_text_col_for_pp

    st.markdown("---")
    st.subheader("Atau pilih file dari database untuk diproses (ulang):")
    try:
        c.execute("SELECT id, filename, filepath, original_text_column FROM uploaded_files ORDER BY id DESC")
        db_files = c.fetchall()
    except sqlite3.Error as e:
        st.error(f"Tidak dapat mengakses database: {e}")
        db_files = []


    if not db_files and data_to_load_for_preprocessing is None:
        st.warning("Tidak ada file di database atau data belum dikonfirmasi dari unggahan. Silakan unggah dan konfirmasi data terlebih dahulu di menu '1. Upload Data'.")
    else:
        # Buat opsi untuk selectbox hanya jika db_files tidak kosong
        if db_files:
            file_options_from_db = {f"{rec[1]} (ID: {rec[0]}) - Kolom Teks Asli: {rec[3] or 'Belum Ditentukan'}": rec for rec in db_files}
            # Tambahkan opsi "Pilih..." hanya jika ada file lain
            selectbox_options = ["Pilih file dari database..."] + list(file_options_from_db.keys())
            
            selected_db_file_key = st.selectbox(
                "Pilih file dari database:",
                selectbox_options,
                key="preprocess_db_select"
            )

            if selected_db_file_key != "Pilih file dari database...":
                selected_record = file_options_from_db[selected_db_file_key]
                db_id, filename_db, filepath_db, text_col_db = selected_record
                
                # Tawarkan tombol proses jika file dari DB dipilih, hanya jika data_to_load_for_preprocessing belum diset dari unggahan baru
                if data_to_load_for_preprocessing is None: # Hanya tampilkan tombol ini jika belum ada data dari upload terakhir yg dipilih
                    if st.button(f"Gunakan File dari DB: '{filename_db}' untuk Preprocessing", key=f"process_db_{db_id}"):
                        df_from_db = load_df_from_path(filepath_db, filename_db)
                        if df_from_db is not None and not df_from_db.empty:
                            data_to_load_for_preprocessing = df_from_db
                            current_text_col_for_pp = text_col_db if text_col_db else (df_from_db.columns[0] if len(df_from_db.columns) > 0 else None) # Fallback
                            source_name = filename_db
                            # Simpan nama kolom teks yang akan diproses dari sumber ini
                            st.session_state.selected_text_col_preprocessing = current_text_col_for_pp
                        else:
                            st.error(f"Gagal memuat file '{filename_db}' dari database atau file kosong.")
        elif data_to_load_for_preprocessing is None: # Jika db_files kosong dan tidak ada data dari upload
            st.warning("Tidak ada file di database untuk dipilih.")


    # Bagian inti preprocessing jika data sudah siap
    if data_to_load_for_preprocessing is not None and current_text_col_for_pp:
        df_original = data_to_load_for_preprocessing
        text_column_to_process = current_text_col_for_pp # Ini adalah nama kolom teks asli

        if text_column_to_process not in df_original.columns:
            st.error(f"Kolom teks asli '{text_column_to_process}' tidak ditemukan dalam data sumber '{source_name}'. Kolom tersedia: {', '.join(df_original.columns.tolist())}")
        else:
            st.markdown(f"Akan memproses kolom **'{text_column_to_process}'** dari file **'{source_name}'**.")
            st.write("Contoh data sebelum preprocessing (5 baris pertama):")
            st.dataframe(df_original[[text_column_to_process]].head())

            # Simpan nama kolom teks asli yang benar-benar akan diproses ke session_state
            # Ini penting untuk referensi di menu selanjutnya
            st.session_state.selected_text_col_preprocessing = text_column_to_process

            if st.button("Mulai Proses Cleaning & Preprocessing Teks", key="start_text_preprocessing_btn"):
                with st.spinner(f"Melakukan preprocessing pada {len(df_original)} baris. Mohon tunggu..."):
                    df_processed_output = df_original.copy()
                    # Pastikan kolom yang diproses ada dan adalah string
                    df_processed_output[text_column_to_process] = df_processed_output[text_column_to_process].astype(str)
                    df_processed_output['processed_text'] = preprocess_text_series_st(df_processed_output[text_column_to_process], stemmer)

                # Simpan hasil ke session state
                st.session_state.processed_df_info = {
                    'original_filename': source_name,
                    'df': df_processed_output,
                    'original_text_column_name': text_column_to_process # Simpan nama kolom teks asli
                }
                st.success("Preprocessing teks selesai!")
                st.subheader("Data Setelah Preprocessing (5 baris pertama):")
                st.dataframe(df_processed_output[[text_column_to_process, 'processed_text']].head())
                st.info(f"Jumlah baris data yang diproses: {len(df_processed_output)}. Silakan lanjutkan ke 'Pelatihan Model SVM'.")
                
                # Membersihkan state upload_file_info agar tidak otomatis terpilih lagi jika pengguna kembali ke menu ini
                # kecuali jika mereka memulai dari Upload Data lagi.
                st.session_state.uploaded_file_info = None 
    
    elif (st.session_state.get('uploaded_file_info') or db_files):
         st.info("Pilih data yang akan diproses dengan menekan tombol 'Gunakan Data Ini...' atau 'Gunakan File dari DB...'.")


elif choice == "3. Pelatihan Model SVM (Pseudo-Labeling)":
    st.header("🏋️ 3. Pelatihan Model SVM (Pseudo-Labeling)")

    if st.session_state.processed_df_info is None:
        st.warning("Data belum diproses. Silakan lakukan preprocessing di menu '2. Preprocessing Data'.")
        st.stop()
    if sentiment_analyzer_pipeline is None: # Menggunakan variabel global yang sudah dicek di sidebar
        st.error("Pipeline model pre-trained sentiment tidak tersedia (gagal dimuat saat awal). Pelatihan tidak dapat dilanjutkan.")
        st.stop()

    processed_info_train = st.session_state.processed_df_info
    df_processed_full = processed_info_train['df']
    original_filename_train = processed_info_train['original_filename']
    # Mengambil nama kolom teks asli yang digunakan saat preprocessing
    original_text_col_name_for_display = processed_info_train.get('original_text_column_name', 'Teks Asli')


    st.info(f"Data yang digunakan untuk pelatihan berasal dari file '{original_filename_train}' yang sudah diproses (kolom 'processed_text').")
    st.write("Contoh data 'processed_text' yang akan digunakan (5 baris pertama):")
    st.dataframe(df_processed_full[['processed_text']].head())

    # Parameter Pelatihan
    st.subheader("Parameter Pelatihan SVM")
    train_subset_ratio = st.slider("Persentase data untuk di-pseudo-label dan training SVM:", 0.05, 1.0, 0.20, 0.05, key="train_subset_ratio_slider", help="Rasio dari data yang sudah diproses untuk dijadikan data latih setelah pseudo-labeling.")
    max_features_tfidf = st.number_input("Jumlah Fitur Maksimum (TF-IDF):", min_value=100, max_value=10000, value=3000, step=100, key="max_feat_tfidf")
    svm_kernel_train = st.selectbox("Kernel SVM:", options=['linear', 'rbf', 'poly'], index=0, key="svm_kernel_train_select")
    svm_c_train = st.number_input("Parameter C (Regularization) SVM:", min_value=0.01, value=1.0, step=0.1, format="%.2f", key="svm_c_train_input")


    if st.button("🚀 Mulai Pelatihan SVM dengan Pseudo-Labels", key="start_svm_pseudo_train_btn"):
        if 'processed_text' not in df_processed_full.columns or df_processed_full['processed_text'].isnull().all():
            st.error("Kolom 'processed_text' tidak ada atau kosong. Pastikan preprocessing berhasil.")
            st.stop()
        
        min_data_for_training = 10 # Minimal data untuk dibagi dan dilatih
        if len(df_processed_full) < min_data_for_training:
             st.error(f"Data yang diproses terlalu sedikit (kurang dari {min_data_for_training} baris). Tidak cukup untuk pelatihan.")
             st.stop()

        with st.spinner(f"Memilih {train_subset_ratio*100:.0f}% data, melakukan pseudo-labeling, dan melatih SVM... Ini mungkin memakan waktu."):
            
            n_total_samples = len(df_processed_full)
            n_train_samples = max(5, int(n_total_samples * train_subset_ratio)) # Minimal 5 sampel untuk latih
            if n_train_samples >= n_total_samples and n_total_samples > 0: # Jika rasio 100% atau lebih
                n_train_samples = n_total_samples # Gunakan semua data
            
            # Ambil subset untuk training
            df_train_subset = df_processed_full.sample(n=n_train_samples, random_state=42)
            st.write(f"Jumlah data untuk pseudo-labeling & training SVM: {len(df_train_subset)} dari {n_total_samples} total data yang diproses.")

            # 2. Pseudo-labeling
            texts_for_pseudo_labeling = df_train_subset['processed_text'].tolist()
            pseudo_labels_list = []
            labeling_errors_count = 0

            progress_bar_pseudo = st.progress(0, text="Pseudo-labeling teks...")
            for i, text_to_label in enumerate(texts_for_pseudo_labeling):
                if pd.isna(text_to_label) or not str(text_to_label).strip():
                    pseudo_labels_list.append("neutral") # Default untuk teks kosong/NaN
                else:
                    try:
                        # Pastikan teks adalah string tunggal, bukan list atau lainnya
                        result = sentiment_analyzer_pipeline(str(text_to_label))
                        pseudo_labels_list.append(result[0]['label'].lower()) # e.g. POSITIVE -> positive
                    except Exception as e_label:
                        # st.warning(f"Error pseudo-labeling teks pendek: '{str(text_to_label)[:30]}...' -> {e_label}. Diberi label 'neutral'.")
                        labeling_errors_count += 1
                        pseudo_labels_list.append("neutral") # Default jika ada error
                progress_bar_pseudo.progress((i + 1) / len(texts_for_pseudo_labeling), text=f"Pseudo-labeling teks... {i+1}/{len(texts_for_pseudo_labeling)}")
            progress_bar_pseudo.empty()

            if labeling_errors_count > 0:
                st.warning(f"{labeling_errors_count} dari {len(texts_for_pseudo_labeling)} teks gagal di-pseudo-label dengan benar dan diberi label 'neutral'.")

            # Assign pseudo-labels ke DataFrame subset pelatihan
            # Penting: pastikan index df_train_subset masih sesuai jika ada modifikasi sebelumnya
            df_train_subset = df_train_subset.copy() # Hindari SettingWithCopyWarning
            df_train_subset.loc[:, 'pseudo_sentiment'] = pseudo_labels_list
            
            st.write("Contoh data setelah pseudo-labeling (5 baris pertama):")
            st.dataframe(df_train_subset[['processed_text', 'pseudo_sentiment']].head())

            # 3. Encode Pseudo-Labels
            current_pseudo_label_encoder = LabelEncoder()
            df_train_subset.loc[:, 'pseudo_sentiment_encoded'] = current_pseudo_label_encoder.fit_transform(df_train_subset['pseudo_sentiment'])
            st.session_state.pseudo_label_encoder = current_pseudo_label_encoder # Simpan encoder
            
            st.write("Mapping Label (dari Pseudo-Labeling) ke Angka:")
            label_mapping = dict(zip(st.session_state.pseudo_label_encoder.classes_, st.session_state.pseudo_label_encoder.transform(st.session_state.pseudo_label_encoder.classes_)))
            st.json(label_mapping)

            # 4. TF-IDF Feature Extraction (FIT HANYA pada subset pelatihan ini)
            current_tfidf_vectorizer = TfidfVectorizer(max_features=max_features_tfidf, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
            X_train_for_svm = current_tfidf_vectorizer.fit_transform(df_train_subset['processed_text'].fillna('')) # fillna untuk safety
            y_train_for_svm = df_train_subset['pseudo_sentiment_encoded']
            st.session_state.tfidf_vectorizer = current_tfidf_vectorizer # Simpan vectorizer yang sudah di-fit

            # 5. Train SVM Model
            current_svm_model = SVC(kernel=svm_kernel_train, C=svm_c_train, probability=True, random_state=42, class_weight='balanced')
            current_svm_model.fit(X_train_for_svm, y_train_for_svm)
            st.session_state.svm_model = current_svm_model # Simpan model yang sudah dilatih

            st.success("Model SVM berhasil dilatih dengan pseudo-label!")

            # Evaluasi model SVM pada data training itu sendiri (terhadap pseudo-label)
            y_pred_on_pseudo_train = st.session_state.svm_model.predict(X_train_for_svm)
            accuracy_on_pseudo_train = accuracy_score(y_train_for_svm, y_pred_on_pseudo_train)
            
            # Pastikan labels dan target_names konsisten untuk classification_report
            unique_labels_encoded = sorted(y_train_for_svm.unique())
            unique_target_names = st.session_state.pseudo_label_encoder.inverse_transform(unique_labels_encoded)

            report_on_pseudo_train_dict = classification_report(
                y_train_for_svm, 
                y_pred_on_pseudo_train,
                labels=unique_labels_encoded,
                target_names=unique_target_names,
                output_dict=True, 
                zero_division=0
            )
            
            st.session_state.svm_training_data = {
                'df_train_pseudo_labeled': df_train_subset.copy(), # Simpan df yang digunakan untuk training
                'original_text_column_used': original_text_col_name_for_display, # Nama kolom teks asli
                'accuracy_on_pseudo_train_set': accuracy_on_pseudo_train,
                'report_on_pseudo_train_set': report_on_pseudo_train_dict
            }
            st.subheader("Evaluasi Kinerja Model SVM (pada Data Training yang di-Pseudo-Label):")
            st.metric("Akurasi pada Data Training (vs Pseudo-Labels)", f"{accuracy_on_pseudo_train:.4f}")
            st.text("Laporan Klasifikasi Detail pada Data Training (vs Pseudo-Labels):")
            st.dataframe(pd.DataFrame(report_on_pseudo_train_dict).transpose().style.format("{:.2f}"))
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

    # Data yang akan diklasifikasi adalah seluruh data yang sudah diproses
    data_to_classify_info = st.session_state.processed_df_info
    df_processed_for_classification = data_to_classify_info['df']
    original_filename_for_classification = data_to_classify_info['original_filename']
    # Mengambil nama kolom teks asli yang digunakan saat preprocessing
    original_text_col_name_for_display_classify = data_to_classify_info.get('original_text_column_name', 'Teks Asli')


    st.info(f"Akan dilakukan klasifikasi sentimen pada seluruh data dari file '{original_filename_for_classification}' (yang telah melalui preprocessing) menggunakan model SVM yang telah dilatih.")
    st.write("Contoh data 'processed_text' yang akan diklasifikasi (5 baris pertama):")
    st.dataframe(df_processed_for_classification[['processed_text']].head())

    if st.button("🔎 Mulai Klasifikasi Sentimen dengan SVM", key="start_svm_full_classification_btn"):
        if 'processed_text' not in df_processed_for_classification.columns or df_processed_for_classification['processed_text'].isnull().all():
            st.error("Kolom 'processed_text' tidak ditemukan atau seluruhnya kosong pada data yang akan diklasifikasi. Pastikan preprocessing berhasil.")
            st.stop()

        with st.spinner("Mengaplikasikan TF-IDF dan memprediksi sentimen dengan SVM pada seluruh data..."):
            df_classified_output = df_processed_for_classification.copy()

            # 1. TF-IDF Transform (menggunakan vectorizer yang sudah di-FIT dari tahap pelatihan)
            texts_for_svm_classification = df_classified_output['processed_text'].fillna('') # Handle NaN untuk safety
            X_features_for_classification = st.session_state.tfidf_vectorizer.transform(texts_for_svm_classification)

            # 2. SVM Predict
            svm_predictions_encoded_output = st.session_state.svm_model.predict(X_features_for_classification)

            # 3. Decode Predictions ke Label Teks
            svm_predictions_text_output = st.session_state.pseudo_label_encoder.inverse_transform(svm_predictions_encoded_output)

            df_classified_output['sentiment_svm_prediction'] = svm_predictions_text_output
            
            # Simpan hasil klasifikasi ke session state
            st.session_state.classified_df_info = {
                'original_filename': original_filename_for_classification,
                'df': df_classified_output,
                'original_text_column_name': original_text_col_name_for_display_classify 
            }
        
        st.success("Klasifikasi sentimen dengan SVM pada seluruh data telah selesai!")
        st.subheader("Contoh Hasil Klasifikasi SVM (5 baris pertama):")
        
        display_cols_for_classify_preview = [original_text_col_name_for_display_classify, 'processed_text', 'sentiment_svm_prediction']
        # Filter kolom yang benar-benar ada
        display_cols_for_classify_preview = [col for col in display_cols_for_classify_preview if col in df_classified_output.columns]
        
        st.dataframe(df_classified_output[display_cols_for_classify_preview].head())
        st.info("Hasil klasifikasi lengkap dapat dilihat dan diunduh di menu '5. Hasil Analisis Sentimen'.")


elif choice == "5. Hasil Analisis Sentimen":
    st.header("📈 5. Hasil Analisis Sentimen (dari Klasifikasi SVM)")

    if not st.session_state.get('classified_df_info'):
        st.warning("Belum ada data yang diklasifikasi menggunakan SVM. Silakan jalankan proses klasifikasi di menu '4. Klasifikasi Sentimen dengan SVM' terlebih dahulu.")
        st.stop()

    final_results_info = st.session_state.classified_df_info
    df_final_svm_results = final_results_info['df']
    original_filename_final_results = final_results_info['original_filename']
    original_text_col_name_final_display = final_results_info.get('original_text_column_name', 'Teks Asli Tidak Diketahui')

    st.subheader(f"Menampilkan Hasil Analisis Sentimen untuk File: '{original_filename_final_results}'")

    if 'sentiment_svm_prediction' in df_final_svm_results.columns:
        # Menyiapkan kolom untuk ditampilkan
        cols_to_display_in_final_table = []
        if original_text_col_name_final_display in df_final_svm_results.columns:
            cols_to_display_in_final_table.append(original_text_col_name_final_display)
        elif st.session_state.get('selected_text_col_preprocessing') and st.session_state.selected_text_col_preprocessing in df_final_svm_results.columns:
            # Fallback ke nama kolom dari preprocessing jika ada
            cols_to_display_in_final_table.append(st.session_state.selected_text_col_preprocessing)
        
        if 'processed_text' in df_final_svm_results.columns:
             cols_to_display_in_final_table.append('processed_text')
        cols_to_display_in_final_table.append('sentiment_svm_prediction')
        
        # Filter hanya kolom yang benar-benar ada
        cols_to_display_in_final_table = [col for col in cols_to_display_in_final_table if col in df_final_svm_results.columns]

        st.subheader("Tabel Lengkap Hasil Klasifikasi Sentimen SVM:")
        if cols_to_display_in_final_table:
            st.dataframe(df_final_svm_results[cols_to_display_in_final_table], height=400) # Tambahkan height agar bisa discroll
        else:
            st.warning("Tidak ada kolom yang valid untuk ditampilkan. Periksa kembali alur data.")


        # Tombol Unduh Hasil
        @st.cache_data 
        def convert_df_for_download(df_to_convert, columns_to_include):
            if columns_to_include:
                return df_to_convert[columns_to_include].to_csv(index=False).encode('utf-8')
            return df_to_convert.to_csv(index=False).encode('utf-8') # Fallback jika kolom tidak spesifik

        if cols_to_display_in_final_table:
            csv_for_download = convert_df_for_download(df_final_svm_results, cols_to_display_in_final_table)
            st.download_button(
                label="📥 Unduh Semua Hasil Klasifikasi (CSV)",
                data=csv_for_download,
                file_name=f"hasil_klasifikasi_svm_{original_filename_final_results.split('.')[0]}.csv",
                mime='text/csv',
                key="download_all_classified_svm_results_csv"
            )
        st.markdown("---")
        st.subheader("Distribusi Sentimen Hasil Klasifikasi SVM pada Seluruh Data:")
        sentiment_svm_counts = df_final_svm_results['sentiment_svm_prediction'].value_counts()

        if not sentiment_svm_counts.empty:
            # Visualisasi dengan st.bar_chart
            st.bar_chart(sentiment_svm_counts)

            # Visualisasi dengan Matplotlib/Seaborn untuk kustomisasi lebih
            try:
                fig_dist, ax_dist = plt.subplots(figsize=(8, 5))
                # Gunakan palette yang konsisten atau menarik
                palette_colors = sns.color_palette("viridis", len(sentiment_svm_counts))
                sns.countplot(
                    x='sentiment_svm_prediction', 
                    data=df_final_svm_results, 
                    order=sentiment_svm_counts.index, # Urutkan berdasarkan frekuensi
                    ax=ax_dist, 
                    palette=palette_colors
                )
                ax_dist.set_title('Distribusi Sentimen (Prediksi SVM pada Seluruh Data)')
                ax_dist.set_xlabel('Sentimen')
                ax_dist.set_ylabel('Jumlah Komentar')
                # Tambahkan label angka di atas bar
                for p in ax_dist.patches:
                    ax_dist.annotate(f'{int(p.get_height())}', 
                                     (p.get_x() + p.get_width() / 2., p.get_height()),
                                     ha='center', va='center', 
                                     xytext=(0, 5), 
                                     textcoords='offset points', fontsize=9)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout() # Atur layout agar tidak terpotong
                st.pyplot(fig_dist)
            except Exception as e_plot:
                st.error(f"Gagal membuat plot distribusi sentimen detail: {e_plot}")
        else:
            st.info("Tidak ada data sentimen yang valid untuk divisualisasikan dari hasil klasifikasi SVM.")
    else:
        st.error("Kolom 'sentiment_svm_prediction' tidak ditemukan dalam data hasil. Proses klasifikasi mungkin belum berhasil atau tidak lengkap.")

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.info("Aplikasi Analisis Sentimen v2.1\n\n(SVM dengan Pseudo-Labeling)")

# Untuk menutup koneksi database saat aplikasi berhenti (opsional, Streamlit biasanya menangani ini)
# def close_db_connection():
#     if conn:
#         conn.close()
# import atexit
# atexit.register(close_db_connection)