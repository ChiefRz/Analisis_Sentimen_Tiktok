import streamlit as st
import pandas as pd

# Pengaturan sidebar
st.sidebar.title("Menu")
st.sidebar.write("Upload Data")
st.sidebar.write("Preprocessing Data")
st.sidebar.write("Hasil Analisis")

# Judul utama
st.title("Pick your file here:")

# Upload file
uploaded_file = st.file_uploader("Select your files here:", type=['csv', 'xlsx', 'txt'], label_visibility="collapsed")

# Tombol Process
if st.button("Process"):
    if uploaded_file is not None:
        # Proses file di sini, misalnya membaca data
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            data = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.txt'):
            data = pd.read_csv(uploaded_file, sep="\t")  # Assuming tab-separated for txt files
        st.success("File berhasil diproses!")
        # Tampilkan tabel hasil
        st.dataframe(data)
    else:
        st.warning("Silakan unggah file terlebih dahulu.")

# Optional: Display section for preprocessing and analysis results
st.sidebar.header("Additional Options")
# Here you can add more options for preprocessing or analysis if needed
