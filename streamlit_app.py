import streamlit as st
import pandas as pd
import sqlite3
import os
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer
import nltk

# Download NLTK resources (uncomment if running for the first time)
nltk.download('punkt')
nltk.download('stopwords')

# Create a connection to the SQLite database (it will be created if it doesn't exist)
conn = sqlite3.connect('data_files.db')
c = conn.cursor()

# Create a table for storing file data if it doesn't already exist
c.execute('''
          CREATE TABLE IF NOT EXISTS files (
              id INTEGER PRIMARY KEY,
              filename TEXT,
              filepath TEXT
          )
          ''')
conn.commit()

# Sidebar Navigation
st.sidebar.title("Menu")
menu_options = ["Upload Data", "Preprocessing Data", "Hasil Analisis"]
choice = st.sidebar.radio("Select an option", menu_options)

# Upload Data Section
if choice == "Upload Data":
    uploaded_file = st.file_uploader("Upload your files here:", type=['csv', 'xlsx', 'txt'])
    
    if st.button("Process"):
        if uploaded_file is not None:
            file_path = f"uploads/{uploaded_file.name}"  # Define where to store the uploaded files
            os.makedirs('uploads', exist_ok=True)  # Create directory if it doesn't exist
            
            try:
                # Save the file to the uploads directory
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Store file metadata in the database
                c.execute("INSERT INTO files (filename, filepath) VALUES (?, ?)",
                          (uploaded_file.name, file_path))
                conn.commit()

                # Process the uploaded file
                if uploaded_file.name.endswith('.csv'):
                    data = pd.read_csv(file_path)
                elif uploaded_file.name.endswith('.xlsx'):
                    data = pd.read_excel(file_path)
                elif uploaded_file.name.endswith('.txt'):
                    data = pd.read_csv(file_path, sep="\t")
                
                if data.empty:
                    st.error("The uploaded file is empty.")
                else:
                    st.success("File berhasil diproses!")
                    st.dataframe(data)
                    
            except pd.errors.EmptyDataError:
                st.error("File is empty or not properly formatted.")
            except Exception as e:
                st.error(f"An error occurred: {e}")

# Preprocessing Data Section
elif choice == "Preprocessing Data":
    st.write("Pick your file here:")
    
    c.execute("SELECT * FROM files")
    files = c.fetchall()
    
    if files:
        options = {file[1]: file[2] for file in files}  # Create a mapping of filename to filepath
        selected_file_name = st.selectbox("Select a file:", options.keys())
        
        if st.button("Process"):
            selected_file_path = options[selected_file_name]  # Get the file path for the selected file
            try:
                # Load the selected file
                if selected_file_name.endswith('.csv'):
                    data = pd.read_csv(selected_file_path)
                elif selected_file_name.endswith('.xlsx'):
                    data = pd.read_excel(selected_file_path)
                elif selected_file_name.endswith('.txt'):
                    data = pd.read_csv(selected_file_path, sep="\t")
                
                # Preprocessing steps
                if 'text' in data.columns:
                    # Fungsi untuk preprocessing teks
                    def preprocess_text(text):
                        # Lowercasing
                        text = text.lower()
                        # Menghapus tanda baca
                        text = text.translate(str.maketrans('', '', string.punctuation))
                        # Tokenisasi
                        tokens = word_tokenize(text)
                        # Menghapus kata henti
                        stop_words = set(stopwords.words('indonesian'))  # Ganti dengan 'english' jika menggunakan bahasa Inggris
                        tokens = [word for word in tokens if word not in stop_words]
                        return ' '.join(tokens)

                    # Terapkan preprocessing pada kolom 'text'
                    data['processed_text'] = data['text'].apply(preprocess_text)
                    
                    # Tokenisasi menggunakan IndoBERT
                    tokenizer = BertTokenizer.from_pretrained('indobert-base-uncased')
                    data['tokenized_text'] = data['processed_text'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))
                    
                    st.write("Data Setelah Diproses:")
                    st.dataframe(data[['text', 'processed_text', 'tokenized_text']])

                    # Menampilkan statistik dasar
                    st.write("Statistik Dasar dari Data yang Diproses:")
                    st.dataframe(data['processed_text'])