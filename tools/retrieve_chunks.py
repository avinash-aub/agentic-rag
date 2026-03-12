from vectorstore.vector_store import get_vector_store
from typing import List
from langchain_core.documents import Document
def get_retriever(collection_name: str):
    '''
    Returns a retriever from the vector store.
    '''
    vector_store = get_vector_store(collection_name)
    if not vector_store:
        raise Exception(f"Vector store for collection {collection_name} not found")
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5}
    )


def retrieve_chunks(collection_name: str, query: str) -> List[Document]:
    '''
    Retrieves chunks from the vector store.
    '''
    retriever = get_retriever(collection_name)
    chunks = retriever.invoke(query)
    return chunks