from typing import TypedDict, List, Optional
from langchain_core.documents import Document
from pydantic import BaseModel, Field

class AgentState(TypedDict):
    query: str
    initial_answer: str
    retrieved_chunks: List[Document]
    is_relevant: bool
    final_answer: str
    data_source: str
    transparency_note: Optional[str]


class RetrievalGrade(BaseModel):
    is_relevant: bool = Field(description="Are the retrieved chunks relevant to the query?")
    reasoning: str = Field(description="Brief reason for the grade")