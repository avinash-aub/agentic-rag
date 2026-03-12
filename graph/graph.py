from langgraph.graph import StateGraph, START, END
from graph.state import AgentState
from graph.nodes import (
    generate_node,
    retrieve_node,
    grader_node,
    alignment_node,
    fallback_node
)

def relevance_router(state: AgentState) -> str:
    '''
    Routes to alignment_node if chunks are relevant,
    otherwise routes to fallback_node.
    '''
    if state["is_relevant"]:
        return "relevant"
    return "not_relevant"


def build_graph():
    '''
    Assembles and compiles the LangGraph agent.
    '''
    graph = StateGraph(AgentState)

    # add nodes
    graph.add_node("generate",  generate_node)
    graph.add_node("retrieve",  retrieve_node)
    graph.add_node("grader",    grader_node)
    graph.add_node("alignment", alignment_node)
    graph.add_node("fallback",  fallback_node)

    # fixed edges
    graph.add_edge(START,       "generate")
    graph.add_edge("generate",  "retrieve")
    graph.add_edge("retrieve",  "grader")

    # conditional edge
    graph.add_conditional_edges(
        "grader",
        relevance_router,
        {
            "relevant":     "alignment",
            "not_relevant": "fallback"
        }
    )

    graph.add_edge("alignment", END)
    graph.add_edge("fallback",  END)

    return graph.compile()


# compiled graph instance
rag_graph = build_graph()
