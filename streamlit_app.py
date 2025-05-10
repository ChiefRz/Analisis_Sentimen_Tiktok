import streamlit as st
import pandas as pd

# Sidebar Navigation
st.sidebar.title("Menu")
menu_options = ["Upload Data", "Preprocessing Data", "Hasil Analisis"]
choice = st.sidebar.radio("Select an option", menu_options)

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

# Preprocessing Data Section
elif choice == "Preprocessing Data":
    st.write("Select a file to process:")
    uploaded_file = st.file_uploader("Select your files here:", type=['csv', 'xlsx', 'txt'])

    if st.button("Process"):
        if uploaded_file is not None:
            # Preprocessing logic here
            # You can add your preprocessing code
            st.success("File ready for preprocessing!")
            st.write("You can implement your preprocessing logic here.")  # Placeholder for preprocessing logic
            # For example, display basic information about the data:
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                data = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.txt'):
                data = pd.read_csv(uploaded_file, sep="\t")
                
            st.dataframe(data.describe())  # Displaying basic statistics

# Placeholder for output
st.write("Output will be displayed here")
