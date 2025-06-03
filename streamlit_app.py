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
from sklearn.pipeline import Pipeline

# --- NLTK Resource Downloads ---
try:
    word_tokenize("test")
except LookupError:
    nltk.download('punkt')
try:
    stopwords.words('indonesian')
except LookupError:
    nltk.download('stopwords')
# User included punkt_tab, let's ensure it's handled if specifically needed,
# otherwise 'punkt' usually covers tokenization.
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt_tab')
    except Exception as e:
        st.sidebar.warning("Gagal mengunduh 'punkt_tab'. Fitur tokenisasi standar ('punkt') akan digunakan.")


# --- Database Setup ---
DB_NAME = 'data_files_svm_app.db' # Use a new DB name or clear old one
conn = sqlite3.connect(DB_NAME, check_same_thread=False) # Added check_same_thread=False for Streamlit
c = conn.cursor()
c.execute('''
          CREATE TABLE IF NOT EXISTS files (
              id INTEGER PRIMARY KEY,
              filename TEXT UNIQUE,
              filepath TEXT,
              is_processed BOOLEAN DEFAULT FALSE, /* Renamed for clarity */
              has_sentiment_labels BOOLEAN DEFAULT FALSE,
              original_text_column TEXT, /* Store original text column name */
              label_column_name TEXT /* Store detected label column name */
          )
          ''')
conn.commit()

# --- Sastrawi Stemmer Initialization ---
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# --- Inisialisasi Session State ---
if 'app_data' not in st.session_state: # General container for app related data
    st.session_state.app_data = {
        "uploaded_file_info": None,
        "data_to_preprocess_path": None,
        "data_to_preprocess_df": None,
        "preprocessed_data_path": None,
        "preprocessed_data_df": None,
        "data_for_analysis_path": None,
        "data_for_analysis_df": None,
        "X_train": None, "X_test": None, "y_train": None, "y_test": None,
        "model_pipeline": None,
        "y_pred": None,
        "accuracy": None,
        "classification_report_dict": None,
        "confusion_matrix_data": None,
        "text_column_for_analysis": None,
        "label_column_for_analysis": None,
    }

# --- Helper Functions ---
@st.cache_data(ttl=3600) # Cache for 1 hour
def load_data_from_path(file_path): # Renamed to avoid conflict
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path, on_bad_lines='skip')
    elif file_path.endswith('.xlsx'):
        return pd.read_excel(file_path)
    elif file_path.endswith('.txt'):
        return pd.read_csv(file_path, sep="\t", on_bad_lines='skip')
    return None

@st.cache_data(ttl=3600)
def _preprocess_text_sastrawi_cached(text, _sastrawi_stemmer): # Pass stemmer explicitly for caching
    if not isinstance(text, str):
        return "", []
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)
    text = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F900-\U0001F9FF\U0001F1E0-\U0001F1FF\u2600-\u26FF\u2700-\u27BF]+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = text.strip()
    tokens = word_tokenize(text)
    stemmed_tokens_sastrawi = [_sastrawi_stemmer.stem(token) for token in tokens]
    stop_words_list = set(stopwords.words('indonesian'))
    filtered_tokens = [word for word in stemmed_tokens_sastrawi if word not in stop_words_list and len(word) > 2]
    return ' '.join(filtered_tokens), filtered_tokens

# Wrapper function to call the cached version with the global stemmer
def preprocess_text_sastrawi(text):
    return _preprocess_text_sastrawi_cached(text, stemmer)


# --- Judul Aplikasi ---
st.title("📱 Aplikasi Analisis Sentimen (TF-IDF & SVM)")
st.write("""
Aplikasi ini melakukan analisis sentimen pada teks menggunakan TF-IDF untuk ekstraksi fitur
dan Support Vector Machine (SVM) sebagai algoritma klasifikasi.
Gunakan menu di sidebar untuk navigasi.
""")

# --- Pilihan Menu ---
menu_options = ["1. Upload Data", "2. Preprocessing Data", "3. Processing Data (TF-IDF & SVM)", "4. Visualisasi Hasil"]
choice = st.sidebar.selectbox("Pilih Menu", menu_options)

# --- Garis Pemisah Antar Bagian ---
st.markdown("---")

# --- Konten Menu ---

if choice == "1. Upload Data":
    st.header("📤 1. Upload Data (CSV, XLSX, TXT)")
    uploaded_file = st.file_uploader("Pilih file:", type=['csv', 'xlsx', 'txt'])

    if uploaded_file is not None:
        if st.button("💾 Proses & Simpan File"):
            filename = uploaded_file.name
            safe_filename = "".join(c if c.isalnum() or c in ['.', '_'] else '_' for c in filename)
            file_path = os.path.join("uploads", safe_filename)
            os.makedirs('uploads', exist_ok=True)

            try:
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                data = load_data_from_path(file_path) # Use renamed helper

                if data is None or data.empty:
                    st.error("File yang diunggah kosong atau format tidak dikenal.")
                else:
                    st.session_state.app_data["uploaded_file_info"] = {
                        "filename": safe_filename,
                        "filepath": file_path
                    }
                    st.success(f"File '{safe_filename}' berhasil diunggah!")
                    st.write("**Pratinjau Data:**")
                    st.dataframe(data.head())
                    st.write(f"**Jumlah Baris:** {data.shape[0]}, **Jumlah Kolom:** {data.shape[1]}")
                    st.info(f"Kolom yang terdeteksi: {', '.join(data.columns.tolist())}")

                    detected_text_col = None
                    common_text_cols = ['text', 'teks', 'review', 'tweet', 'komentar', 'ulasan']
                    for col_name in common_text_cols:
                        if col_name in data.columns:
                            detected_text_col = col_name
                            break
                    
                    potential_label_cols = ['sentiment', 'label', 'sentimen', 'kelas']
                    detected_label_col = None
                    for col in potential_label_cols:
                        if col in data.columns:
                            detected_label_col = col
                            break

                    has_sentiment_col_flag = bool(detected_label_col)

                    try:
                        c.execute("INSERT INTO files (filename, filepath, has_sentiment_labels, original_text_column, label_column_name) VALUES (?, ?, ?, ?, ?)",
                                  (safe_filename, file_path, has_sentiment_col_flag, detected_text_col, detected_label_col))
                        conn.commit()
                        st.info(f"Metadata file '{safe_filename}' disimpan ke database.")
                        if detected_text_col:
                            st.success(f"Kolom teks potensial terdeteksi: '{detected_text_col}'")
                        else:
                            st.warning(f"Tidak ada kolom teks umum ({', '.join(common_text_cols)}) yang terdeteksi. Anda perlu menentukannya pada tahap preprocessing jika berbeda.")
                        if has_sentiment_col_flag:
                            st.success(f"Kolom label sentimen potensial terdeteksi: '{detected_label_col}'")
                        else:
                            st.warning(f"Tidak ada kolom label sentimen umum ({', '.join(potential_label_cols)}) yang terdeteksi. Analisis SVM memerlukan kolom label.")

                    except sqlite3.IntegrityError:
                        st.warning(f"File dengan nama '{safe_filename}' sudah ada di database. Jika ini file baru, ubah namanya. Jika file lama, Anda bisa memprosesnya ulang di menu Preprocessing.")
                    except Exception as db_e:
                        st.error(f"Gagal menyimpan metadata ke database: {db_e}")

            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses file: {e}")
    else:
        st.info("Silakan upload file untuk memulai.")

    st.subheader("📦 Daftar File di Database")
    try:
        files_in_db = pd.read_sql_query("SELECT id, filename, filepath, is_processed, has_sentiment_labels, original_text_column, label_column_name FROM files", conn)
        if not files_in_db.empty:
            st.dataframe(files_in_db, use_container_width=True)
        else:
            st.info("Belum ada file di database.")
    except Exception as e:
        st.error(f"Gagal memuat daftar file dari database: {e}")


elif choice == "2. Preprocessing Data":
    st.header("🧹 2. Preprocessing Data Teks")

    c.execute("SELECT id, filename, filepath, original_text_column FROM files WHERE is_processed = FALSE")
    unprocessed_files = c.fetchall()

    if not unprocessed_files:
        st.warning("Tidak ada file baru yang perlu diproses atau semua file sudah diproses. Unggah file baru terlebih dahulu di menu '1. Upload Data'.")
    else:
        options = {f"{file_rec[1]} (ID: {file_rec[0]})": file_rec for file_rec in unprocessed_files}
        selected_file_display_name = st.selectbox("Pilih file untuk preprocessing:", options.keys())

        if selected_file_display_name: # Check if a file is selected
            selected_file_record = options[selected_file_display_name]
            default_text_col = selected_file_record[3] if selected_file_record[3] else "text"
            text_column_to_process = st.text_input("Masukkan nama kolom yang berisi teks (jika berbeda dari deteksi awal):", default_text_col)

            if st.button("🔄 Mulai Preprocessing"):
                if not text_column_to_process:
                    st.error("Silakan masukkan nama kolom teks.")
                else:
                    file_id, filename, filepath, _ = selected_file_record
                    st.session_state.app_data["data_to_preprocess_path"] = filepath

                    try:
                        data = load_data_from_path(filepath) # Use renamed helper
                        if data is None or data.empty:
                            st.error("Gagal memuat data atau file kosong.")
                            st.stop()

                        st.session_state.app_data["data_to_preprocess_df"] = data.copy()

                        if text_column_to_process not in data.columns:
                            st.error(f"Kolom '{text_column_to_process}' tidak ditemukan dalam file. Kolom tersedia: {', '.join(data.columns.tolist())}")
                            st.stop()

                        st.write("**Data Asli (Sebelum Preprocessing):**")
                        st.dataframe(data[[text_column_to_process]].head())

                        with st.spinner("Sedang melakukan preprocessing... Ini mungkin memakan waktu."):
                            preprocessing_results = data[text_column_to_process].apply(lambda x: preprocess_text_sastrawi(str(x)))
                            data['processed_text'] = preprocessing_results.apply(lambda x: x[0])
                            data['tokens_sastrawi'] = preprocessing_results.apply(lambda x: x[1])

                        st.session_state.app_data["preprocessed_data_df"] = data.copy()

                        st.subheader("Data Setelah Diproses:")
                        st.dataframe(data[[text_column_to_process, 'processed_text', 'tokens_sastrawi']].head())

                        original_filename_parts = os.path.splitext(os.path.basename(filepath))
                        processed_file_name = f"{original_filename_parts[0]}_processed.csv"
                        processed_file_path = os.path.join("uploads", processed_file_name)

                        data.to_csv(processed_file_path, index=False)
                        st.session_state.app_data["preprocessed_data_path"] = processed_file_path

                        c.execute("UPDATE files SET is_processed = ?, original_text_column = ? WHERE id = ?", (True, text_column_to_process, file_id))
                        
                        # Retrieve original label column name and has_sentiment_labels status for the new processed file entry
                        c.execute("SELECT has_sentiment_labels, label_column_name FROM files WHERE id = ?", (file_id,))
                        original_file_meta = c.fetchone()
                        has_sentiment_original = original_file_meta[0] if original_file_meta else False
                        label_col_original = original_file_meta[1] if original_file_meta else None

                        c.execute("SELECT id FROM files WHERE filename = ?", (processed_file_name,))
                        existing_processed_file = c.fetchone()
                        if existing_processed_file:
                            c.execute("UPDATE files SET filepath = ?, is_processed = ?, has_sentiment_labels = ?, original_text_column = ?, label_column_name = ? WHERE id = ?",
                                      (processed_file_path, True, has_sentiment_original, 'processed_text', label_col_original, existing_processed_file[0]))
                        else:
                            c.execute("INSERT INTO files (filename, filepath, is_processed, has_sentiment_labels, original_text_column, label_column_name) VALUES (?, ?, ?, ?, ?, ?)",
                                      (processed_file_name, processed_file_path, True, has_sentiment_original, 'processed_text', label_col_original))
                        conn.commit()

                        st.success(f"Data yang telah diproses disimpan sebagai '{processed_file_name}'.")
                        st.info("Anda sekarang dapat menggunakan file ini di menu 'Processing Data'.")

                    except Exception as e:
                        st.error(f"Terjadi kesalahan saat preprocessing: {e}")
                        st.error(f"Detail: {type(e).__name__}, {e.args}")

elif choice == "3. Processing Data (TF-IDF & SVM)":
    st.header("⚙️ 3. Processing Data (TF-IDF & SVM)")

    c.execute("SELECT id, filename, filepath, is_processed, has_sentiment_labels, original_text_column, label_column_name FROM files")
    all_files_records = c.fetchall()

    if not all_files_records:
        st.warning("Tidak ada file yang tersedia. Silakan unggah atau proses file terlebih dahulu.")
        st.stop()

    file_options_analysis = {}
    for record in all_files_records:
        id_f, name_f, path_f, processed_f, has_label_f, orig_txt_col, lbl_col = record
        display_name = f"{name_f} (ID: {id_f})"
        tags = []
        if processed_f: tags.append("Telah Diproses")
        else: tags.append("Asli (Belum Diproses)")
        if has_label_f and lbl_col: tags.append(f"Label: '{lbl_col}'")
        elif has_label_f: tags.append("Ada Label (Nama Kolom ?)")
        else: tags.append("Tanpa Label Terdeteksi")

        display_name_with_tag = f"{display_name} [{', '.join(tags)}]"
        file_options_analysis[display_name_with_tag] = record

    selected_file_display_name_analysis = st.selectbox(
        "Pilih file untuk analisis sentimen:",
        file_options_analysis.keys()
    )
    
    text_col_svm = None
    label_col_svm = None
    data_svm_loaded_successfully = False

    if selected_file_display_name_analysis:
        selected_record_for_analysis = file_options_analysis[selected_file_display_name_analysis]
        file_id_an, filename_an, filepath_an, is_proc_an, has_lbl_an, orig_txt_col_an, lbl_col_an = selected_record_for_analysis
        
        # --- Otomatisasi Penentuan Kolom Teks ---
        data_svm = load_data_from_path(filepath_an) # Load data once for column checks

        if data_svm is None or data_svm.empty:
            st.error(f"Gagal memuat data atau file '{filename_an}' kosong.")
        else:
            data_svm_loaded_successfully = True
            if is_proc_an:
                if 'processed_text' in data_svm.columns:
                    text_col_svm = 'processed_text'
                    st.info(f"Menggunakan kolom teks yang telah diproses: '{text_col_svm}' dari file '{filename_an}'.")
                else:
                    st.error(f"File '{filename_an}' ditandai telah diproses, tetapi kolom 'processed_text' tidak ditemukan.")
            else: # File belum diproses
                if orig_txt_col_an and orig_txt_col_an in data_svm.columns:
                    text_col_svm = orig_txt_col_an
                    st.info(f"Menggunakan kolom teks dari database: '{text_col_svm}' dari file '{filename_an}'.")
                else:
                    common_text_cols_analysis = ['text', 'teks', 'review', 'tweet', 'komentar', 'ulasan']
                    for col_name in common_text_cols_analysis:
                        if col_name in data_svm.columns:
                            text_col_svm = col_name
                            st.info(f"Kolom teks terdeteksi otomatis: '{text_col_svm}' dari file '{filename_an}'.")
                            break
                if not text_col_svm:
                    st.error(f"Tidak dapat menentukan kolom teks secara otomatis untuk file '{filename_an}'. Pastikan file memiliki kolom teks yang sesuai atau perbarui metadata di database.")
            
            if not is_proc_an and text_col_svm:
                 st.warning(f"File '{filename_an}' belum diproses. Untuk hasil optimal, disarankan untuk memproses file terlebih dahulu melalui menu '2. Preprocessing Data'.")


            # --- Otomatisasi Penentuan Kolom Label ---
            if has_lbl_an and lbl_col_an and lbl_col_an in data_svm.columns:
                label_col_svm = lbl_col_an
                st.info(f"Menggunakan kolom label dari database: '{label_col_svm}'.")
            elif has_lbl_an: # has_sentiment_labels is true but column name might be missing or incorrect in DB
                potential_label_cols_analysis = ['sentiment', 'label', 'sentimen', 'kelas']
                for col in potential_label_cols_analysis:
                    if col in data_svm.columns:
                        label_col_svm = col
                        st.info(f"Kolom label terdeteksi otomatis: '{label_col_svm}'.")
                        # Optionally, update DB here if lbl_col_an was missing/wrong
                        # c.execute("UPDATE files SET label_column_name = ? WHERE id = ?", (label_col_svm, file_id_an))
                        # conn.commit()
                        break
                if not label_col_svm: # Still not found even if DB said has_label
                     st.error(f"Metadata file '{filename_an}' mengindikasikan ada label, tetapi kolom label ({', '.join(potential_label_cols_analysis)}) tidak dapat ditemukan di file. Periksa file atau metadata.")
            elif not has_lbl_an and not label_col_svm : # No label indicated by DB, try one last detection
                potential_label_cols_analysis = ['sentiment', 'label', 'sentimen', 'kelas']
                for col in potential_label_cols_analysis:
                    if col in data_svm.columns:
                        label_col_svm = col
                        st.warning(f"Kolom label terdeteksi otomatis: '{label_col_svm}'. Database tidak mengindikasikan adanya label sebelumnya untuk file ini.")
                        # Optionally, update DB here
                        # c.execute("UPDATE files SET has_sentiment_labels = TRUE, label_column_name = ? WHERE id = ?", (label_col_svm, file_id_an))
                        # conn.commit()
                        break

            if not label_col_svm:
                st.error(f"Kolom label tidak dapat ditentukan untuk file '{filename_an}'. Analisis SVM memerlukan kolom label. Pastikan file Anda memiliki kolom label yang sesuai atau perbarui metadata.")

    # Default SVM parameters
    default_test_size = 0.2
    default_kernel = 'linear'
    default_c = 1.0
    default_gamma = 'scale'

    with st.expander("🔧 Atur Parameter Model SVM (Opsional)"):
        test_size_svm = st.slider("Ukuran data testing", 0.1, 0.5, default_test_size, 0.05, key="svm_test_size_adv")
        svm_kernel = st.selectbox("Kernel SVM", ('linear', 'rbf', 'poly', 'sigmoid'), index=['linear', 'rbf', 'poly', 'sigmoid'].index(default_kernel), key="svm_kernel_select_adv")
        svm_c = st.number_input("Parameter C (Regularization)", value=default_c, min_value=0.01, format="%.2f", key="svm_c_val_adv")
        if svm_kernel in ['rbf', 'poly', 'sigmoid']:
            svm_gamma = st.select_slider("Parameter Gamma", options=['scale', 'auto', 0.001, 0.01, 0.1, 1], value=default_gamma, key="svm_gamma_val_adv")
        else:
            svm_gamma = default_gamma # Default for linear
    
    # Use defaults if expander is not used or keep advanced settings
    # For simplicity in this refactor, we'll directly use the widget values,
    # which will hold defaults if expander isn't touched.
    current_test_size = st.session_state.get("svm_test_size_adv", default_test_size)
    current_kernel = st.session_state.get("svm_kernel_select_adv", default_kernel)
    current_c = st.session_state.get("svm_c_val_adv", default_c)
    if current_kernel in ['rbf', 'poly', 'sigmoid']:
        current_gamma = st.session_state.get("svm_gamma_val_adv", default_gamma)
    else:
        current_gamma = default_gamma


    if st.button("🚀 Latih Model SVM & Prediksi"):
        if not data_svm_loaded_successfully:
            st.error("Data belum berhasil dimuat. Pilih file yang valid.")
            st.stop()
        if not text_col_svm or not label_col_svm:
            st.error("Kolom teks atau label belum dapat ditentukan. Periksa pesan di atas.")
            st.stop()

        try:
            # data_svm is already loaded for column checks
            st.session_state.app_data["data_for_analysis_path"] = filepath_an
            st.session_state.app_data["data_for_analysis_df"] = data_svm.copy() # Use the already loaded df
            st.session_state.app_data["text_column_for_analysis"] = text_col_svm
            st.session_state.app_data["label_column_for_analysis"] = label_col_svm

            # Ensure columns exist (double check, though previous logic should catch it)
            if text_col_svm not in data_svm.columns or label_col_svm not in data_svm.columns:
                 st.error(f"Error kritis: Kolom teks ('{text_col_svm}') atau label ('{label_col_svm}') tidak ditemukan setelah pengecekan akhir.")
                 st.stop()

            data_svm.dropna(subset=[text_col_svm, label_col_svm], inplace=True)
            if data_svm.empty:
                st.error("Data kosong setelah menghapus baris dengan nilai NaN pada kolom teks atau label.")
                st.stop()

            X = data_svm[text_col_svm].astype(str)
            y = data_svm[label_col_svm].astype(str)

            if len(X) < 2 or len(y.unique()) < 2:
                st.error(f"Data tidak cukup ({len(X)} sampel) atau hanya ada satu ({len(y.unique())}) kelas sentimen di kolom '{label_col_svm}'. SVM memerlukan minimal 2 sampel dan 2 kelas.")
                st.stop()

            with st.spinner("Melatih model SVM dengan TF-IDF..."):
                try:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=current_test_size, random_state=42, stratify=y)
                except ValueError as e_split:
                    st.warning(f"Stratifikasi gagal ({e_split}), mencoba split tanpa stratifikasi.")
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=current_test_size, random_state=42)

                st.session_state.app_data["X_train"], st.session_state.app_data["X_test"] = X_train, X_test
                st.session_state.app_data["y_train"], st.session_state.app_data["y_test"] = y_train, y_test

                model_pipeline = Pipeline([
                    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
                    ('svm', SVC(kernel=current_kernel, C=current_c, gamma=current_gamma, probability=True, random_state=42))
                ])
                model_pipeline.fit(X_train, y_train)
                y_pred = model_pipeline.predict(X_test)

                st.session_state.app_data["model_pipeline"] = model_pipeline
                st.session_state.app_data["y_pred"] = y_pred

                accuracy = accuracy_score(y_test, y_pred)
                # Ensure labels for classification report are from the actual unique values in y_test and y_pred combined
                # or from y.unique() if that's more representative of all possible labels. Model's classes_ is best.
                unique_labels_for_report = sorted(list(set(y_train.unique()) | set(y_test.unique()))) #model_pipeline.classes_
                
                report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0, labels=unique_labels_for_report, target_names=unique_labels_for_report)
                cm_data = confusion_matrix(y_test, y_pred, labels=unique_labels_for_report)

                st.session_state.app_data["accuracy"] = accuracy
                st.session_state.app_data["classification_report_dict"] = report_dict
                st.session_state.app_data["confusion_matrix_data"] = cm_data

            st.success("Model SVM berhasil dilatih!")
            st.metric(label="Akurasi Model pada Data Uji", value=f"{accuracy:.4f}")
            st.info("Hasil evaluasi detail tersedia di menu '4. Visualisasi Hasil'.")

            st.subheader("🧪 Uji Model dengan Teks Baru")
            new_text_input = st.text_area("Masukkan teks untuk diprediksi sentimennya:")
            if st.button("Prediksi Teks Baru"):
                if new_text_input and st.session_state.app_data["model_pipeline"]:
                    processed_new_text, _ = preprocess_text_sastrawi(new_text_input)
                    prediction = st.session_state.app_data["model_pipeline"].predict([processed_new_text])
                    prediction_proba = st.session_state.app_data["model_pipeline"].predict_proba([processed_new_text])

                    st.write(f"**Teks Asli:** {new_text_input}")
                    st.write(f"**Teks Setelah Preprocessing:** {processed_new_text}")
                    st.write(f"**Prediksi Sentimen:** **{prediction[0]}**")
                    st.write("**Probabilitas Prediksi:**")
                    classes_ = st.session_state.app_data["model_pipeline"].classes_
                    for i, class_label in enumerate(classes_):
                        st.write(f"  - {class_label}: {prediction_proba[0][i]:.4f}")
                elif not new_text_input:
                    st.warning("Masukkan teks terlebih dahulu.")
                else:
                    st.warning("Model belum dilatih.")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat processing: {e}")
            st.error(f"Detail Error - Tipe: {type(e).__name__}, Pesan: {e}")


elif choice == "4. Visualisasi Hasil":
    st.header("📊 4. Visualisasi Hasil Analisis Sentimen")

    if st.session_state.app_data.get("model_pipeline") and st.session_state.app_data.get("y_pred") is not None:
        accuracy = st.session_state.app_data["accuracy"]
        report_dict = st.session_state.app_data["classification_report_dict"]
        cm_data = st.session_state.app_data["confusion_matrix_data"]
        y_test_vis = st.session_state.app_data["y_test"]
        y_pred_vis = st.session_state.app_data["y_pred"]
        X_test_vis = st.session_state.app_data["X_test"]
        model_classes = st.session_state.app_data["model_pipeline"].classes_
        label_col_name_vis = st.session_state.app_data.get("label_column_for_analysis", "Label")


        st.metric(label="Akurasi Model pada Data Uji", value=f"{accuracy:.4f}")

        st.subheader("Laporan Klasifikasi:")
        if report_dict:
            report_df = pd.DataFrame(report_dict).transpose()
            st.dataframe(report_df.style.format("{:.2f}"))
        else:
            st.write("Tidak ada laporan klasifikasi untuk ditampilkan.")

        st.subheader("Matriks Konfusi (Confusion Matrix):")
        if cm_data is not None:
            fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
            # Use model_classes which are derived from training data y values for consistency
            sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues',
                        xticklabels=model_classes, yticklabels=model_classes, ax=ax_cm)
            ax_cm.set_xlabel("Predicted Label")
            ax_cm.set_ylabel("True Label")
            ax_cm.set_title('Confusion Matrix')
            st.pyplot(fig_cm)
        else:
            st.write("Tidak ada confusion matrix untuk ditampilkan.")

        st.subheader("Contoh Hasil Prediksi pada Data Uji (10 Sampel Teratas):")
        if X_test_vis is not None and y_test_vis is not None and y_pred_vis is not None:
            results_df_vis = pd.DataFrame({
                'Teks Uji': X_test_vis.head(10).values,
                f'Label Asli ({label_col_name_vis})': y_test_vis.head(10).values,
                'Prediksi Sentimen SVM': y_pred_vis[:10]
            })
            st.dataframe(results_df_vis, use_container_width=True)

        st.subheader("Distribusi Sentimen (Hasil Prediksi pada Data Uji):")
        if y_pred_vis is not None:
            fig_dist_pred, ax_dist_pred = plt.subplots(figsize=(10, 6))
            sentiment_counts_pred = pd.Series(y_pred_vis).value_counts().reindex(model_classes, fill_value=0)
            sns.barplot(x=sentiment_counts_pred.index, y=sentiment_counts_pred.values, ax=ax_dist_pred, order=model_classes, palette="viridis")
            ax_dist_pred.set_title('Distribusi Sentimen Hasil Prediksi SVM pada Data Uji')
            ax_dist_pred.set_xlabel('Sentimen')
            ax_dist_pred.set_ylabel('Jumlah')
            plt.xticks(rotation=45)
            st.pyplot(fig_dist_pred)

        st.subheader("Distribusi Sentimen (Label Asli pada Data Uji):")
        if y_test_vis is not None:
            fig_dist_true, ax_dist_true = plt.subplots(figsize=(10, 6))
            sentiment_counts_true = y_test_vis.value_counts().reindex(model_classes, fill_value=0)
            sns.barplot(x=sentiment_counts_true.index, y=sentiment_counts_true.values, ax=ax_dist_true, order=model_classes, palette="coolwarm")
            ax_dist_true.set_title('Distribusi Sentimen Label Asli pada Data Uji')
            ax_dist_true.set_xlabel('Sentimen')
            ax_dist_true.set_ylabel('Jumlah')
            plt.xticks(rotation=45)
            st.pyplot(fig_dist_true)

    else:
        st.warning("Silakan latih model SVM terlebih dahulu pada menu '3. Processing Data (TF-IDF & SVM)' untuk melihat visualisasi.")

# --- Informasi Tambahan ---
st.sidebar.markdown("---")
st.sidebar.info(
    "Aplikasi Analisis Sentimen v1.2\n\n"
    "Pastikan file data Anda memiliki kolom teks dan kolom label sentimen yang jelas."
)

# --- Close database connection (Streamlit generally handles this on script rerun) ---
# conn.close()