## Agentic RAG

This project is an **agentic RAG pipeline** that:

- **Crawls / fetches** web content.
- **Cleans and chunks** documents using LangChain text splitters.
- **Indexes chunks** into a vector store for retrieval.
- **Runs a graph of nodes** (see `graph/`) to orchestrate retrieval, tool calls, and LLM reasoning.

### Graph overview

The high-level graph used in this repo:

![RAG Graph](rag_graph.png)

At a glance:

- **Ingestion path**: tools in `tools/` pull and split data into semantic chunks.
- **Graph nodes**: `graph/nodes.py` wires together retrieval + LLM steps.
- **Vector store**: `vectorstore/` holds embeddings and supports similarity search.
