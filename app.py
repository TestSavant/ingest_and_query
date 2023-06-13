# Import necessary libraries
import streamlit as st
import subprocess

# Create two columns for Ingest and Chat
col1, col2 = st.beta_columns(2)

# Ingest Column
with col1:
    st.header("Ingest")
    # Document Upload
    uploaded_file = st.file_uploader("Choose a file")
    # Source Directory Input
    source_directory = st.text_input("Enter source directory")
    # Device Type Selection
    device_type = st.selectbox("Select device type", ["cuda", "cpu", "hip"])
    # Index Name Input
    index_name = st.text_input("Enter index name for Pinecone")
    # Ingest Button
    if st.button("Ingest"):
        # Call the ingest.py script with the necessary arguments
        subprocess.call(f"python ingest.py --source_dir {source_directory} --device_type {device_type} --index_name {index_name}", shell=True)

# Chat Column
with col2:
    st.header("Chat")
    # Model Selection
    model_selection = st.selectbox("Select model", ["Model 1", "Model 2", "Model 3"])  # Replace with actual model names
    # Environment Variables Input
    openai_api_key = st.text_input("Enter OpenAI API Key", type="password")
    pinecone_api_key = st.text_input("Enter Pinecone API Key", type="password")
    pinecone_environment = st.text_input("Enter Pinecone Environment", type="password")
    # Pinecone Settings Input
    pinecone_settings = st.text_input("Enter Pinecone settings")
    # Question Input
    question = st.text_input("Enter a question")
    # Send Button
    if st.button("Send"):
        # Call the query.py script with the necessary arguments
        subprocess.call(f"python query.py --model {model_selection} --openai_api_key {openai_api_key} --pinecone_api_key {pinecone_api_key} --pinecone_environment {pinecone_environment} --pinecone_settings {pinecone_settings} --question {question}", shell=True)
