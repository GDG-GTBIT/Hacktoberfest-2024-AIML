import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import AIMessage, FunctionMessage, HumanMessage
from langchain_core.tools import Tool
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, VectorParams


load_dotenv()


class VectorStore:
    def __init__(self):
        #self.embeddings = CohereEmbeddings(model="embed-english-v3.0")
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        # Enabling monitoring with Langsmith
        # os.environ["LANGCHAIN_TRACING_V2"] = "true"
        # os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        # os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")

        self.collection_name = os.getenv("QDRANT_COLLECTION_NAME", "trashcan")
        self.qdrant_url = os.getenv("QDRANT_HOST")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if not self.qdrant_url or not self.qdrant_api_key:
            raise ValueError(
                "QDRANT_URL and QDRANT_API_KEY environment variables must be set"
            )

        self.qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_HOST"), api_key=os.getenv("QDRANT_API_KEY")
        )
        
        try:
            self.db_client = self.get_or_create_collection()
        except Exception as e:
            print(f"Error setting up vector store: {e}")
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=20,
            length_function=len,
        )

    def verify_qdrant_connection(self):
        try:
            self.qdrant_client.get_collections()
            print("Successfully connected to Qdrant server.")
            return True
        except UnexpectedResponse as e:
            print(f"Error connecting to Qdrant server: {e}")
            print(f"Qdrant URL: {self.qdrant_url}")
            print(
                "Please check your QDRANT_URL and QDRANT_API_KEY environment variables."
            )
            return False
        except Exception as e:
            print(f"Unexpected error when connecting to Qdrant server: {e}")
            return False

    def get_or_create_collection(self) -> QdrantVectorStore:
        try:
            collections = self.qdrant_client.get_collections().collections
            collection_exists = any(
                collection.name == self.collection_name for collection in collections
            )
            if not collection_exists:
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
                )
                print(f"Created new collection: {self.collection_name}")
            return QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=self.collection_name,
                embedding=self.embeddings,
            )
        except UnexpectedResponse as e:
            print(f"Error accessing Qdrant collection: {e}")
            raise

    def chunk_text(self, text: str) -> List[str]:
        return self.text_splitter.split_text(text)
    
    def store_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            chunks = []
            for document in documents:
                text = document.get("text", "")
                chunks.extend(self.chunk_text(text))
            self.db_client.add_texts(chunks)
            return {
                "status": "success",
                "stored_count": len(chunks),
                "original_count": len(documents),
            }
        except Exception as e:
            print(f"Error storing documents: {e}")
            return {"status": "error", "message": str(e)}
