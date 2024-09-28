from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

from pymongo.operations import SearchIndexModel
client = MongoClient(os.getenv("MONGODB_URI"))
collection = client["rag_db"]["embedded_resumes"]

search_index_model = SearchIndexModel(
  definition = {
    "fields": [
      {
        "type": "vector",
        "numDimensions": 384,
        "path": "embedding",
        "similarity": "cosine"
      }
    ]
  },
  name = "resume_index",
  type = "vectorSearch"
)
collection.create_search_index(model=search_index_model)
