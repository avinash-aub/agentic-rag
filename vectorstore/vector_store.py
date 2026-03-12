import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import Distance, VectorParams
from qdrant_client import QdrantClient
from tools.split_chunk_data import split_chunk_data
load_dotenv()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=3072
)

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

def get_or_create_collection(collection_name: str):
    '''
    Creates a new collection in Qdrant.
    '''
    try:
        # Retrieve collection if it exists
        collection = client.get_collection(collection_name)

        # Create collection if it doesn't exist
        if not collection:
            collection = client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=3072,
                    distance=Distance.COSINE
                )
            )
        return collection
    except Exception as e:
        print(f"Error creating collection: {e}")
        return None


def get_vector_store(collection_name: str):
    '''
    Returns a vector store for the given collection.
    '''
    try:
        collection = get_or_create_collection(collection_name)
        if not collection:
            raise Exception(f"Collection {collection_name} not found")
        return QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings
        )
    except Exception as e:
        print(f"Error getting vector store: {e}")
        return None


def add_documents_to_vector_store(collection_name: str, url: str):
    '''
    Adds documents to the vector store for the given collection.
    '''
    try:
        vector_store = get_vector_store(collection_name)
        chunks = split_chunk_data(url)
        vector_store.add_documents(chunks)
        print(f"Documents added to vector store for {url}")
        return True
    except Exception as e:
        print(f"Error adding documents to vector store: {e}")
        return False