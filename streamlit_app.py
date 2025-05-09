import streamlit as st
import pandas as pd

# Pengaturan sidebar
st.sidebar.title("Menu")
st.sidebar.write("Upload Data")
st.sidebar.write("Preprocessing Data")
st.sidebar.write("Hasil Analisis")

# Judul utama
st.title("Upload your file here:")

# Upload file
uploaded_file = st.file_uploader("Drag and drop file here", type=['csv', 'xlsx', 'txt'], label_visibility="collapsed")

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
