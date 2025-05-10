import streamlit as st
import pandas as pd

# Sidebar Navigation
st.sidebar.title("Menu")
menu_options = ["Upload Data", "Preprocessing Data", "Hasil Analisis"]
choice = st.sidebar.radio("Select an option", menu_options)

# Main title
st.title("Pick your file here:")

# Main content based on selected option
if choice == "Upload Data":
    st.write("Pick your file here:")
    uploaded_file = st.file_uploader("Select your files here:", type=['csv', 'xlsx'])
    if uploaded_file is not None:
        # You can add code to read and process the uploaded file here
        st.success("File uploaded successfully!")
        
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
