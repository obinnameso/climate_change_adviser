import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
# from langchain_groq import GroqEmbeddings
# from langchain_groq import ChatGroq
# from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

## load document
loader = PyPDFLoader("data/terdoo_report.pdf")
document = loader.load() 

## split into chunks 
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(document)
print(f"created {len(texts)} chunks")

######## test to replace openai embedding 
# embeddings = GroqEmbeddings(
#     model="text-embedding-ada-002",  # Specify your desired model
#     api_key=os.environ["GROQ_API_KEY"]
# )
######
embeddings = OpenAIEmbeddings(base_url="https://api.groq.com/openai/v1",
    openai_api_type=os.environ.get("GROQ_API_KEY"))

# EMBEDDING_MODEL_NAME = "thenlper/gte-small"
# embeddings = HuggingFaceEmbeddings(
#     model_name=EMBEDDING_MODEL_NAME,
#     multi_process=True,
#     model_kwargs={"device": "cpu"}
#     # encode_kwargs={"normalize_embeddings": True},  # Set `True` for cosine similarity
# )
# embeddings = GroqEmbeddings(api_key=GROQ_API_KEY)
PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ.get("climate-vectorized"))

print('finished!')

