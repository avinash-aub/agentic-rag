from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
import trafilatura

def load_web_page_data(url: str) -> List[Document]:
    '''
    Loads a web page from the given URL and returns a list of documents.
    Resulted data contains raw/noisy HTML code.
    '''
    loader = WebBaseLoader([url])
    result = loader.load()
    return result

def load_clean_web_page_data(url: str) -> List[Document]:
    '''
    Fetches a web page from the given URL and returns a clean text.
    Resulted data contains clean text without noisy HTML code.
    '''
    downloaded = trafilatura.fetch_url(url)
    clean_text = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=False,
        no_fallback=False
    )
    document = Document(page_content=clean_text, metadata={"source": url})
    return [document]
