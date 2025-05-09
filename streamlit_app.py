import streamlit as st

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
        st.success("File berhasil diproses!")
        # Tampilkan informasi atau hasil analisis dari file jika diperlukan
    else:
        st.warning("Silakan unggah file terlebih dahulu.")
