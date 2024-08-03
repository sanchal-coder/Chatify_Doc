from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from sentence_transformers.cross_encoder.CrossEncoder import CrossEncoder
import textwrap

def load_pdf_data(file_path):
    loader=PyMuPDFLoader(file_path=file_path)
    docs=loader.load()
    return docs
def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks=text_splitter.split_documents(documents=documents)

    return chunks


def load_embedding_model(model_path, normalize_embedding=True):
    return HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={'device':'cpu'},
        encode_kwargs={
            'normalize_embeddings':normalize_embedding
        }
    )

def create_embeddings(chunks, embedding_model, storage_path="vectorstore"):
    vectorstore=FAISS.from_documents(chunks, embedding_model)
    vectorstore.save_local(storage_path)
    return vectorstore


def load_q_chain(retriever,llm,prompt):
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={'prompt':prompt}
    )

def get_response(query,chain):

    response=chain({'query':query})

    wrapped_text=textwrap.fill(response['result'],width=100)
    return wrapped_text
