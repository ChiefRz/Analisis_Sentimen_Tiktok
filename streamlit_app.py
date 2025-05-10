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

# Preprocessing Data Section
elif choice == "Preprocessing Data":
    st.write("Pick your file here:")
    
    # Dropdown to select from previously uploaded files
    if st.session_state.uploaded_files:
        selected_file = st.selectbox("Select a file:", st.session_state.uploaded_files, format_func=lambda x: x.name)
    else:
        st.warning("No files uploaded yet!")
        selected_file = None
    if st.button("Process"):
        if selected_file is not None:
             # Preprocessing logic here
            st.success("File ready for preprocessing!")
            # Read the selected file
            if selected_file.name.endswith('.csv'):
                data = pd.read_csv(selected_file)
            elif selected_file.name.endswith('.xlsx'):
                data = pd.read_excel(selected_file)
            elif selected_file.name.endswith('.txt'):
                data = pd.read_csv(selected_file, sep="\t")
                
            st.dataframe(data.describe())  # Displaying basic statistics
        else:
            st.warning("Please select a file to process.")
            
# Placeholder for output
st.write("Output will be displayed here")
