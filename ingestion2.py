import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
from mistralai import Mistral
import requests
import numpy as np
import faiss
import os
# from llama_index import MistralAI

load_dotenv()

api_key= os.environ.get("mistral_api_key")
if not api_key:
    raise ValueError("Mistral API key is not set. Please set the 'mistral_api_key' environment variable.")

client = Mistral(api_key=api_key)
# Load data
loader = PyPDFLoader("data/terd_report.pdf")
docs = loader.load()
# Split text into chunks 
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
documents = text_splitter.split_documents(docs)
print(f"created {len(documents)} chunks")
# Define the embedding model
embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=api_key)
# Create the vector store 
vector = FAISS.from_documents(documents, embeddings)
# Define a retriever interface
retriever = vector.as_retriever()
# Define LLM
model = ChatMistralAI(mistral_api_key=api_key)
# model = MistralAI(api_key=api_key, model="mistral-medium")
# Define prompt template
prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

# Create a retrieval chain to answer questions
document_chain = create_stuff_documents_chain(model, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)
user_input = input("Enter your question: ")
response = retrieval_chain.invoke({"input": user_input})
if "answer" not in response:
    raise ValueError("No answer was returned by the retrieval chain.")

print(response["answer"])