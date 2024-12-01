from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_mistralai.chat_models import ChatMistralAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
# import os
# from dotenv import load_dotenv

def load_documents(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return docs

def chunk_documents(docs, chunk_size=1000, chunk_overlap=100):
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(docs)

def create_vector_store(documents, model_name='sentence-transformers/all-mpnet-base-v2'):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return FAISS.from_documents(documents, embeddings)

def create_retrieval_chain_with_prompt(vector_store, api_key):
    retriever = vector_store.as_retriever()
    model = ChatMistralAI(mistral_api_key=api_key)
    prompt = ChatPromptTemplate.from_template("""You are a helpful assistant. Answer questions based on the provided context.
If context is insufficient, respond with "I need more context to provide an accurate answer."

<context>
{context}
</context>

Question: {input}

Answer:""")
    document_chain = create_stuff_documents_chain(model, prompt)
    return create_retrieval_chain(retriever, document_chain)