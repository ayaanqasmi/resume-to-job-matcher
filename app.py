from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document  # Import the Document class
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_cohere import CohereEmbeddings
from pymongo import MongoClient
from dotenv import load_dotenv
import os

from utils.extractpdf import extract_text_from_pdf

load_dotenv()

# Directory setup
current_dir = os.path.dirname(os.path.abspath(__file__))
resumes_dir = os.path.join(current_dir, "INFORMATION-TECHNOLOGY")

if not os.path.exists(resumes_dir):
    raise FileNotFoundError(f"The directory {resumes_dir} does not exist. Please check the path.")

# List all PDF files in the directory
resume_files = [f for f in os.listdir(resumes_dir) if f.endswith('.pdf')]

# Load and process documents using PyPDFLoader
documents = []
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500, chunk_overlap=50
)

for resume_file in resume_files:
    file_path = os.path.join(resumes_dir, resume_file)
    documentText = extract_text_from_pdf(file_path)
    
    # Convert the text to a Document object
    document = Document(page_content=documentText, metadata={"source": resume_file})
    
    # Split the document into chunks
    chunks = text_splitter.split_documents([document])  # Pass the document in a list
    
    # Append the chunks with metadata to the documents list
    documents.extend(chunks)



embeddings = CohereEmbeddings(model="embed-english-light-v3.0")

# MongoDB setup
client = MongoClient(os.getenv("MONGODB_URI"))
collection = client["rag_db"]["embedded_resumes"]

# Create MongoDB Atlas Vector Search
vector_store = MongoDBAtlasVectorSearch.from_documents(
    documents=documents,
    embedding=embeddings,
    collection=collection,
    index_name="resume_index"
)

