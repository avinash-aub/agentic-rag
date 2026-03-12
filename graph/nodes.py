from langchain_openai import ChatOpenAI
from graph.state import AgentState, RetrievalGrade
from graph.prompts import generate_prompt, grader_prompt, alignment_prompt
from tools.retrieve_chunks import retrieve_chunks
from dotenv import load_dotenv
import os
load_dotenv()

COLLECTION_NAME = os.getenv("COLLECTION_NAME")

llm = ChatOpenAI(model="gpt-4o",temperature=0.2)
grader_llm = llm.with_structured_output(RetrievalGrade)

def generate_node(state: AgentState) -> AgentState:
    response = generate_prompt | llm
    result = response.invoke({"query": state["query"]})
    return {
        **state,
        "initial_answer": result.content
    }


def retrieve_node(state: AgentState) -> AgentState:
    chunks = retrieve_chunks(COLLECTION_NAME, state["query"])
    return {
        **state,
        "retrieved_chunks": chunks
    }


def grader_node(state: AgentState) -> AgentState:
    chunks_text = "\n\n".join([doc.page_content for doc in state["retrieved_chunks"]])
    grader_chain = grader_prompt | grader_llm
    result = grader_chain.invoke({
        "query": state["query"],
        "chunks": chunks_text
    })
    return {
        **state,
        "is_relevant": result.is_relevant
    }


def alignment_node(state: AgentState) -> AgentState:
    chunks_text = "\n\n".join([doc.page_content for doc in state["retrieved_chunks"]])
    alignment_chain = alignment_prompt | llm
    result = alignment_chain.invoke({
        "query": state["query"],
        "initial_answer": state["initial_answer"],
        "chunks": chunks_text
    })
    return {
        **state,
        "final_answer": result.content,
        "data_source": "rag",
        "transparency_note": None
    }


def fallback_node(state: AgentState) -> AgentState:
    return {
        **state,
        "final_answer": state["initial_answer"],
        "data_source": "llm_knowledge",
        "transparency_note": (
            "⚠️ No relevant data found in knowledge base. "
            "This answer is based on LLM's own knowledge and may not be fully accurate. "
            "Consider adding relevant sources to the knowledge base."
        )
    }