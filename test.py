import os
import streamlit as st
from ChatBot import *
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

# Initialize the language model and embedding model
llm = Ollama(model="llama3.1", temperature=0)
target_folder = "uploaded_document"
os.makedirs(target_folder, exist_ok=True)

embed = load_embedding_model(model_path="all-MiniLM-L6-v2")

# Define the prompt template
template = """
### System:
You are a respectful and honest assistant. You have to answer the user's \
questions using only the context provided to you. If you don't know the answer, \
just say you don't know. Don't try to make up an answer.

### Context:
{context}

### User:
{question}

### Response:
"""
prompt = PromptTemplate.from_template(template)

# Set up Streamlit file uploader
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # Save the uploaded file to the specified folder
    bytes_data = uploaded_file.read()
    file_path = os.path.join(target_folder, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(bytes_data)
    
    # Load and process the PDF document
    docs = load_pdf_data(file_path)  # Pass the file_path to load_pdf_data
    documents = split_docs(documents=docs)
    vectorstore = create_embeddings(documents, embed)
    retriever = vectorstore.as_retriever()

    # Load the question-answering chain
    chain = load_q_chain(retriever, llm, prompt)
    
    # Streamlit form for user interaction
    with st.form("main_form"):
        st.write("Talk With My PDF:")
        result = st.text_input("Please Enter Your Question:")
        submitted = st.form_submit_button("Send")
        if submitted:
            response = get_response(result, chain)
            st.write(response)
