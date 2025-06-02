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

# Scikit-learn for TF-IDF and SVM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline # Optional, but good for chaining

# --- NLTK Resource Downloads ---
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')

# --- Database Setup ---
conn = sqlite3.connect('data_files_svm.db') # Use a new DB name or clear old one
c = conn.cursor()
c.execute('''
          CREATE TABLE IF NOT EXISTS files (
              id INTEGER PRIMARY KEY,
              filename TEXT,
              filepath TEXT,
              has_processed_text BOOLEAN DEFAULT FALSE,
              has_sentiment_labels BOOLEAN DEFAULT FALSE 
          )
          ''')
conn.commit()

# --- Sastrawi Stemmer Initialization ---
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# --- Sidebar Navigation ---
st.sidebar.title("📜 Menu Aplikasi")
menu_options = ["Unggah Data", "Preprocessing Data", "Hasil Analisis (SVM + TF-IDF)"]
choice = st.sidebar.radio("Pilih Opsi:", menu_options)

# --- Global Variables / Session State (Optional but good for more complex apps) ---
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None

# --- Upload Data Section ---
if choice == "Unggah Data":
    st.header("📤 Unggah File Data Anda")
    uploaded_file = st.file_uploader("Pilih file CSV, XLSX, atau TXT:", type=['csv', 'xlsx', 'txt'])
    
    if uploaded_file is not None:
        if st.button("💾 Proses & Simpan File"):
            filename = uploaded_file.name
            # Sanitize filename to prevent directory traversal or invalid characters
            safe_filename = "".join(c if c.isalnum() or c in ['.', '_'] else '_' for c in filename)
            file_path = os.path.join("uploads", safe_filename)
            
            os.makedirs('uploads', exist_ok=True)
            
            try:
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Attempt to read the file to check basic format and columns
                if safe_filename.endswith('.csv'):
                    data = pd.read_csv(file_path)
                elif safe_filename.endswith('.xlsx'):
                    data = pd.read_excel(file_path)
                elif safe_filename.endswith('.txt'):
                    # Assuming tab-separated for .txt, adjust if necessary
                    data = pd.read_csv(file_path, sep="\t", on_bad_lines='skip') 
                
                if data.empty:
                    st.error("File yang diunggah kosong atau format tidak dikenal.")
                else:
                    # Store file metadata in the database
                    # Check if 'text' column exists for future processing
                    has_text_col = 'text' in data.columns
                    # Check if a common sentiment label column exists (can be made more flexible)
                    has_sentiment_col = any(col in data.columns for col in ['sentiment', 'label', 'sentimen'])

                    c.execute("INSERT INTO files (filename, filepath, has_sentiment_labels) VALUES (?, ?, ?)",
                              (safe_filename, file_path, has_sentiment_col))
                    conn.commit()
                    st.success(f"File '{safe_filename}' berhasil diunggah dan disimpan!")
                    st.dataframe(data.head())
                    st.info(f"Kolom yang terdeteksi: {', '.join(data.columns.tolist())}")
                    if not has_text_col:
                        st.warning("Peringatan: Kolom 'text' tidak ditemukan. Preprocessing dan analisis mungkin memerlukan kolom ini.")
                    if has_sentiment_col:
                        st.info("Kolom label sentimen (misalnya 'sentiment', 'label') terdeteksi.")
                    else:
                        st.warning("Peringatan: Tidak ada kolom label sentimen umum yang terdeteksi ('sentiment', 'label', 'sentimen'). Analisis SVM memerlukan kolom label.")


            except pd.errors.EmptyDataError:
                st.error("File kosong atau format tidak sesuai.")
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
    else:
        st.info("Silakan unggah file untuk memulai.")

# --- Preprocessing Data Section ---
elif choice == "Preprocessing Data":
    st.header("🛠️ Preprocessing Data Teks")
    
    c.execute("SELECT id, filename, filepath FROM files WHERE has_processed_text = FALSE") # Only show unprocessed files
    files = c.fetchall()
    
    if not files:
        st.warning("Tidak ada file baru yang perlu diproses atau semua file sudah diproses. Unggah file baru terlebih dahulu.")
    else:
        options = {f"{file[1]} (ID: {file[0]})": (file[0], file[2]) for file in files}
        selected_file_display_name = st.selectbox("Pilih file untuk preprocessing:", options.keys())
        
        text_column = st.text_input("Masukkan nama kolom yang berisi teks (umumnya 'text'):", "text")

        if st.button("🔄 Mulai Preprocessing"):
            if not selected_file_display_name:
                st.warning("Silakan pilih file.")
            elif not text_column:
                st.error("Silakan masukkan nama kolom teks.")
            else:
                selected_file_id, selected_file_path = options[selected_file_display_name]
                
                try:
                    if selected_file_path.endswith('.csv'):
                        data = pd.read_csv(selected_file_path)
                    elif selected_file_path.endswith('.xlsx'):
                        data = pd.read_excel(selected_file_path)
                    elif selected_file_path.endswith('.txt'):
                        data = pd.read_csv(selected_file_path, sep="\t", on_bad_lines='skip')
                    else:
                        st.error("Format file tidak didukung untuk preprocessing ini.")
                        st.stop()
                    
                    st.session_state.data = data.copy() # Store original data

                    if text_column not in data.columns:
                        st.error(f"Kolom '{text_column}' tidak ditemukan dalam file yang dipilih.")
                        st.info(f"Kolom yang tersedia: {', '.join(data.columns.tolist())}")
                        st.stop()

                    # --- Text Preprocessing Function ---
                    @st.cache_data # Cache the expensive stemming process
                    def preprocess_text_sastrawi(text):
                        if not isinstance(text, str):
                            return "" # Return empty string for non-string inputs
                        text = text.lower() # Lowercasing
                        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Remove URLs
                        text = re.sub(r'\@\w+|\#','', text) # Remove mentions and hashtags
                        text = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F900-\U0001F9FF\U0001F1E0-\U0001F1FF\u2600-\u26FF\u2700-\u27BF]+', '', text) # Remove emoticons
                        text = text.translate(str.maketrans('', '', string.punctuation)) # Remove punctuation
                        text = re.sub(r'\d+', '', text) # Remove numbers
                        text = text.strip() # Remove leading/trailing whitespace
                        
                        tokens = word_tokenize(text) # Tokenization
                        
                        # Stemming with Sastrawi
                        stemmed_tokens = [stemmer.stem(token) for token in tokens]
                        
                        stop_words = set(stopwords.words('indonesian'))
                        filtered_tokens = [word for word in stemmed_tokens if word not in stop_words and len(word) > 2] # Remove stopwords and short words
                        return ' '.join(filtered_tokens)

                    with st.spinner("Sedang melakukan preprocessing... Ini mungkin memakan waktu."):
                        data['processed_text'] = data[text_column].apply(preprocess_text_sastrawi)
                    
                    st.session_state.processed_data = data.copy() # Store processed data

                    st.subheader("Data Setelah Diproses:")
                    st.dataframe(data[[text_column, 'processed_text']].head())

                    st.subheader("Statistik Dasar dari Teks yang Diproses:")
                    st.write(data['processed_text'].describe())

                    # Save processed data
                    original_filename_parts = os.path.splitext(os.path.basename(selected_file_path))
                    processed_file_name = f"{original_filename_parts[0]}_processed.csv"
                    processed_file_path = os.path.join("uploads", processed_file_name)
                    
                    # Ensure all original columns are carried over, especially potential label columns
                    data.to_csv(processed_file_path, index=False)

                    # Update database for the original file entry
                    c.execute("UPDATE files SET has_processed_text = TRUE WHERE id = ?", (selected_file_id,))
                    # Add the new processed file to the database
                    has_sentiment_col_processed = any(col in data.columns for col in ['sentiment', 'label', 'sentimen'])
                    c.execute("INSERT INTO files (filename, filepath, has_processed_text, has_sentiment_labels) VALUES (?, ?, ?, ?)",
                              (processed_file_name, processed_file_path, True, has_sentiment_col_processed )) # <--- DIPERBAIKI DI SINI
                    conn.commit()
                    
                    st.success(f"Data yang telah diproses disimpan sebagai '{processed_file_name}' dan metadata diperbarui di database.")
                    st.info("Anda sekarang dapat menggunakan file yang telah diproses ini di bagian 'Hasil Analisis'.")

                except pd.errors.EmptyDataError:
                    st.error("File kosong atau format tidak sesuai.")
                except KeyError as e:
                    st.error(f"Kolom tidak ditemukan: {e}. Pastikan nama kolom teks benar.")
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat preprocessing: {e}")
                    st.error(f"Detail: {type(e).__name__}, {e.args}")


# --- Hasil Analisis (SVM + TF-IDF) Section ---
elif choice == "Hasil Analisis (SVM + TF-IDF)":
    st.header("📊 Hasil Analisis Sentimen (SVM & TF-IDF)")

    # Fetch files that have been processed OR original files that might have labels
    c.execute("SELECT id, filename, filepath, has_processed_text, has_sentiment_labels FROM files")
    all_files_records = c.fetchall()

    if not all_files_records:
        st.warning("Tidak ada file yang tersedia. Silakan unggah dan proses data terlebih dahulu.")
    else:
        # Prioritize processed files for selection
        # Display filename and whether it's processed or has labels
        file_options = {}
        for record in all_files_records:
            id, name, path, processed, labels = record
            display_name = f"{name} (ID: {id})"
            if processed:
                display_name += " [Processed]"
            elif labels: # Original file but has labels
                 display_name += " [Original with Labels]"
            else: # Original file, no labels, not processed (less ideal for SVM)
                 display_name += " [Original]"
            file_options[display_name] = {'id': id, 'path': path, 'is_processed': processed, 'has_labels': labels}

        selected_file_display_name = st.selectbox("Pilih file untuk analisis sentimen:", file_options.keys())
        
        selected_file_info = file_options[selected_file_display_name]
        selected_file_path = selected_file_info['path']

        # Determine the text column to use
        if selected_file_info['is_processed']:
            text_col_default = 'processed_text'
            st.info("File ini tampaknya sudah diproses. Kolom 'processed_text' akan digunakan secara default.")
        else:
            text_col_default = 'text'
            st.warning("File ini belum diproses. Hasil SVM mungkin lebih baik dengan teks yang sudah dibersihkan. Kolom 'text' akan digunakan.")

        col1, col2 = st.columns(2)
        with col1:
            text_column_for_svm = st.text_input("Nama kolom teks untuk TF-IDF:", text_col_default)
        with col2:
            label_column_for_svm = st.text_input("Nama kolom label sentimen (mis: 'sentiment', 'label'):", "sentiment")

        test_size_svm = st.slider("Ukuran Test Set (mis: 0.2 untuk 20%):", 0.1, 0.5, 0.2, 0.05)

        if st.button("🚀 Lakukan Analisis Sentimen dengan SVM"):
            if not selected_file_path:
                st.warning("Silakan pilih file.")
                st.stop()
            if not text_column_for_svm or not label_column_for_svm:
                st.error("Nama kolom teks dan kolom label harus diisi.")
                st.stop()

            try:
                # Load the selected file
                if selected_file_path.endswith('.csv'):
                    data_for_svm = pd.read_csv(selected_file_path)
                elif selected_file_path.endswith('.xlsx'):
                    data_for_svm = pd.read_excel(selected_file_path)
                elif selected_file_path.endswith('.txt'):
                     data_for_svm = pd.read_csv(selected_file_path, sep="\t", on_bad_lines='skip')
                else:
                    st.error("Format file tidak didukung.")
                    st.stop()

                if text_column_for_svm not in data_for_svm.columns:
                    st.error(f"Kolom teks '{text_column_for_svm}' tidak ditemukan dalam file.")
                    st.info(f"Kolom yang tersedia: {', '.join(data_for_svm.columns.tolist())}")
                    st.stop()
                if label_column_for_svm not in data_for_svm.columns:
                    st.error(f"Kolom label '{label_column_for_svm}' tidak ditemukan.")
                    st.info(f"Kolom yang tersedia: {', '.join(data_for_svm.columns.tolist())}")
                    st.stop()

                # Drop rows with NaN in text or label column
                data_for_svm.dropna(subset=[text_column_for_svm, label_column_for_svm], inplace=True)
                
                if data_for_svm.empty:
                    st.error("Data kosong setelah menghapus baris dengan nilai NaN pada kolom teks atau label.")
                    st.stop()

                # Ensure labels are strings for consistent mapping later
                data_for_svm[label_column_for_svm] = data_for_svm[label_column_for_svm].astype(str)
                
                X = data_for_svm[text_column_for_svm].astype(str) # Ensure text is string
                y = data_for_svm[label_column_for_svm]

                # Check if there's enough data and enough classes
                if len(X) < 2 or len(y.unique()) < 2:
                    st.error("Data tidak cukup atau hanya ada satu kelas sentimen. SVM memerlukan setidaknya 2 sampel dan 2 kelas berbeda untuk training.")
                    st.stop()
                
                with st.spinner("Melatih model SVM dengan TF-IDF... Ini mungkin memerlukan waktu."):
                    # Split data
                    try:
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_svm, random_state=42, stratify=y)
                    except ValueError as e:
                         # If stratification fails (e.g. too few samples per class for split)
                        st.warning(f"Stratifikasi gagal: {e}. Mencoba split tanpa stratifikasi.")
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_svm, random_state=42)


                    # Create a pipeline: TF-IDF -> SVM
                    # You can tune parameters of TfidfVectorizer and SVC
                    pipeline_svm = Pipeline([
                        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))), # Consider tuning parameters
                        ('svm', SVC(kernel='linear', C=1, probability=True, random_state=42)) # Linear kernel is often good for text
                    ])

                    pipeline_svm.fit(X_train, y_train)
                    y_pred = pipeline_svm.predict(X_test)

                st.subheader("Evaluasi Model SVM")
                accuracy = accuracy_score(y_test, y_pred)
                st.write(f"🎯 **Akurasi Model:** {accuracy:.4f}")

                st.text("📝 **Laporan Klasifikasi:**")
                report = classification_report(y_test, y_pred, zero_division=0) # Added zero_division
                st.text(report)
                
                # Store predictions for display
                # Create a DataFrame for easier display of test results
                results_df = pd.DataFrame({
                    'Teks Asli (dari X_test)': X_test,
                    f'Label Asli ({label_column_for_svm})': y_test,
                    'Prediksi Sentimen': y_pred
                })
                
                st.subheader("Contoh Hasil Prediksi pada Data Uji:")
                st.dataframe(results_df.head(10))

                st.subheader("📊 Distribusi Sentimen (Hasil Prediksi pada Data Uji)")
                
                # Ensure consistent order for plotting if possible
                unique_labels_sorted = sorted(y.unique())

                fig, ax = plt.subplots(figsize=(10, 6))
                # Count occurrences of each predicted sentiment
                # Use pd.Series on y_pred to use value_counts()
                sentiment_counts_pred = pd.Series(y_pred).value_counts().reindex(unique_labels_sorted, fill_value=0)
                
                sns.barplot(x=sentiment_counts_pred.index, y=sentiment_counts_pred.values, ax=ax, order=unique_labels_sorted, palette="viridis")
                ax.set_title('Distribusi Sentimen Hasil Prediksi SVM')
                ax.set_xlabel('Sentimen')
                ax.set_ylabel('Jumlah')
                plt.xticks(rotation=45)
                st.pyplot(fig)

                # Confusion Matrix
                st.subheader("Matriks Konfusi")
                cm = confusion_matrix(y_test, y_pred, labels=unique_labels_sorted)
                fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=unique_labels_sorted, yticklabels=unique_labels_sorted, ax=ax_cm)
                ax_cm.set_xlabel('Predicted Labels')
                ax_cm.set_ylabel('True Labels')
                ax_cm.set_title('Confusion Matrix')
                st.pyplot(fig_cm)


            except pd.errors.EmptyDataError:
                st.error("File yang dipilih kosong atau format tidak sesuai.")
            except FileNotFoundError:
                st.error(f"File tidak ditemukan di path: {selected_file_path}. Harap periksa database atau unggah ulang.")
            except KeyError as e:
                st.error(f"Kolom '{e}' tidak ditemukan. Pastikan nama kolom teks dan label benar.")
            except ValueError as e:
                st.error(f"ValueError: {e}. Ini bisa terjadi jika data tidak cukup untuk splitting atau kelas tidak seimbang.")
            except Exception as e:
                st.error(f"Terjadi kesalahan saat analisis: {e}")
                st.error(f"Detail Kesalahan: Type: {type(e).__name__}, Arguments: {e.args}")

# --- Close database connection when app stops (optional, Streamlit handles it) ---
# conn.close() # Usually not explicitly needed in Streamlit as scripts rerun