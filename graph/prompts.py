from langchain_core.prompts import ChatPromptTemplate

generate_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant. Answer the user's query 
     based on your own knowledge as accurately as possible."""),
    ("human", "{query}")
])

grader_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a grader assessing relevance of retrieved chunks to a user query.
     If chunks contain information relevant to the query, grade as relevant.
     Give a binary score: relevant or not relevant."""),
    ("human", "Query: {query}\n\nRetrieved Chunks: {chunks}")
])

alignment_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant. You have two things:
     1. Your initial answer based on your own knowledge
     2. Ground truth retrieved from a verified knowledge base
     
     Refine your initial answer using the ground truth.
     Stay faithful to the ground truth, correct any mistakes in your initial answer."""),
    ("human", """Query: {query}
Initial Answer: {initial_answer}
Ground Truth Chunks:
{chunks}
Provide a refined, grounded answer.""")
])