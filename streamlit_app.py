import streamlit as st
import pandas as pd

# Sidebar Navigation
st.sidebar.title("Menu")
menu_options = ["Upload Data", "Preprocessing Data", "Hasil Analisis"]
choice = st.sidebar.radio("Select an option", menu_options)

if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

# Upload Data Section
if choice == "Upload Data":
    uploaded_file = st.file_uploader("Upload your files here:", type=['csv', 'xlsx', 'txt'])
    
    if st.button("Process"):
        if uploaded_file is not None:
            # Process the uploaded file
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                data = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.txt'):
                data = pd.read_csv(uploaded_file, sep="\t")  # Assuming tab-separated for txt files
            
            st.success("File berhasil diproses!")
            st.dataframe(data)
            st.session_state.uploaded_files.append(uploaded_file)

            try:
                if selected_file.name.endswith('.csv'):
                    data = pd.read_csv(selected_file)
                elif selected_file.name.endswith('.xlsx'):
                    data = pd.read_excel(selected_file)
                elif selected_file.name.endswith('.txt'):
                    data = pd.read_csv(selected_file, sep="\t")
                
                if 'text' in data.columns:
                    processed_data = data[['text']]
                    st.write("Data Setelah Diproses:")
                    st.dataframe(processed_data)
                else:
                    st.error("Kolom 'text' tidak ditemukan dalam data.")
                
                st.dataframe(data.describe())  # Displaying basic statistics
            except pd.errors.EmptyDataError:
                st.error("The selected file is empty or not properly formatted.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please select a file to process.")
