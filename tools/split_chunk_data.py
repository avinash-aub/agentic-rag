from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tools.extract_data import load_clean_web_page_data

def split_chunk_data(url: str) -> List[Document]:
    '''
    Loads clean text from the given URL and splits it into overlapping text chunks as Document objects.
    '''
    document = load_clean_web_page_data(url)
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=200, chunk_overlap=50)
    chunks = splitter.split_documents(document)
    return chunks