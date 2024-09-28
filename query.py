from langchain_cohere import CohereEmbeddings
from langchain_cohere.llms import Cohere
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()



# Create your index model, then create the search index
embeddings = CohereEmbeddings(model="embed-english-light-v3.0")

client = MongoClient(os.getenv("MONGODB_URI"))
collection = client["rag_db"]["embedded_resumes"]

vector_store = MongoDBAtlasVectorSearch(
    embedding = embeddings,
    collection = collection,
    index_name = "resume_index"
)



if vector_store:
    print("Vector store is initialized.")
else:
    print("Vector store initialization failed.")

# Check for document count in the collection
doc_count = collection.count_documents({})
print(f"Number of documents in the collection: {doc_count}")

retriever = vector_store.as_retriever(
    search_type="similarity"
)
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)


prompt = PromptTemplate.from_template("""
   Find the most suitable resume based on the following Job description in terms of experience. Explain why. Give answer in json format like:src:"",pageContent:"", explanation:"", src and pageContent being from metadata and the actual document and explanation being your reasoning.
   Job Description: {question}
   Context: {context}
""")
rag_chain = (
   { "context": retriever, "question": RunnablePassthrough()}
   | prompt
   | llm
   | StrOutputParser()
)
question = (
    "Ayaan Qasmi"
)


try:
    results = rag_chain.invoke(question)
    if results:
        print(results)
    else:
        print("No results found.")
except Exception as e:
    print(f"An error occurred: {e}")
