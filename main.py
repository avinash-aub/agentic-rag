from graph.graph import rag_graph

def save_graph_image():
    """
    Saves the graph as a PNG image.
    """
    png_data = rag_graph.get_graph().draw_mermaid_png()

    with open("rag_graph.png", "wb") as f:
        f.write(png_data)

    print("Graph saved as rag_graph.png")


def run_agent(query: str):
    '''
    Runs the agentic RAG pipeline for a given query.
    '''
    initial_state = {
        "query": query,
        "initial_answer": "",
        "retrieved_chunks": [],
        "is_relevant": False,
        "final_answer": "",
        "data_source": "",
        "transparency_note": None
    }

    result = rag_graph.invoke(initial_state)

    print("\n" + "="*50)
    print(f"Query: {result['query']}")
    print("="*50)
    print(f"\n📝 Final Answer:\n{result['final_answer']}")
    print(f"\n📊 Data Source: {result['data_source']}")
    if result['transparency_note']:
        print(f"\n{result['transparency_note']}")
    print("="*50)

    return result