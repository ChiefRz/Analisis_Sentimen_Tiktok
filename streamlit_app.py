import streamlit as st
import pandas as pd

# Sidebar Navigation
st.sidebar.title("Menu")
menu_options = ["Upload Data", "Preprocessing Data", "Hasil Analisis"]
choice = st.sidebar.radio("Select an option", menu_options)

# Main content based on selected option
if choice == "Upload Data":
    uploaded_file = st.file_uploader("Upload your files here:", type=['csv', 'xlsx'])
    st.button("Process"): if uploaded_file is not None: 
    # Proses file di sini, misalnya membaca data 
    if uploaded_file.name.endswith('.csv'): 
        data = pd.read_csv(uploaded_file) 
    elif uploaded_file.name.endswith('.xlsx'): 
        data = pd.read_excel(uploaded_file) 
    elif uploaded_file.name.endswith('.txt'): 
        data = pd.read_csv(uploaded_file, sep="\t") 
    
    # Assuming tab-separated for txt files 
    st.success("File berhasil diproses!") 
    # Tampilkan tabel hasil 
    st.dataframe(data)  
        
        
elif choice == "Preprocessing Data":
    st.write("Select a file to process:")
    uploaded_file = st.file_uploader("Select your files here:", type=['csv', 'xlsx'])
    if uploaded_file is not None:
        # You can add code to preprocess the data here
        st.success("File ready for preprocessing!")
    
# Process button
if st.button("Process"):
    st.write("Processing...")
    # You can add your processing logic here

# Placeholder for output
st.write("Output will be displayed here")
