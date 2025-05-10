import streamlit as st
import pandas as pd

# Sidebar configuration
st.sidebar.title("Menu")
st.sidebar.radio("Navigate", options=["Upload Data", "Preprocessing Data", "Hasil Analisis"])

# Main title
st.title("Pick your file here:")

# File uploader
uploaded_file = st.file_uploader("Select your files here:", type=['csv', 'xlsx', 'txt'], label_visibility="collapsed")

# Process button
if st.button("Process"):
    if uploaded_file is not None:
        # Process the file here, e.g., read data
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            data = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.txt'):
            data = pd.read_csv(uploaded_file, sep="\t")  # Assuming tab-separated for txt files
        
        st.success("File berhasil diproses!")
        # Display the resulting dataframe
        st.dataframe(data)
    else:
        st.warning("Silakan unggah file terlebih dahulu.")
