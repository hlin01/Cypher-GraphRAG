import os
from dotenv import load_dotenv

from langchain_community.chains.graph_qa.memgraph import MemgraphQAChain
from langchain_community.graphs import MemgraphGraph
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()
os.environ["OPENAI_API_KEY"]

url = os.environ.get("MEMGRAPH_URI", "bolt://localhost:7687")
username = os.environ.get("MEMGRAPH_USERNAME", "")
password = os.environ.get("MEMGRAPH_PASSWORD", "")

# initialize memgraph connection (run construct_kg.py first)
graph = MemgraphGraph(
    url=url, username=username, password=password, refresh_schema=False
)

llm = ChatOpenAI(temperature=0, model_name="gpt-4o")

cypher_template = """
You are an expert in crafting precise and executable Cypher queries for a Memgraph database based on a given schema.

Instructions:
- Generate ONE valid Cypher query that directly answers the user's natural language question.
- Use only the node labels, relationship types, and property names provided in the schema.
- The query must be executable on a Memgraph database and use Memgraph MAGE procedures when applicable (do not use Neo4j APOC procedures).
- Do not include any text other than the Cypher query (no explanations, context, or questions).
- Do not include the text 'cypher' before the query.

Schema:
{schema}

Below are examples of questions and their corresponding Cypher queries. Use them to help you generate valid and correct queries.

Example 1:
Question: Who did Cl meet?
Cypher query: MATCH (a:Person {{id: 'Cl'}})-[:MET_WITH]->(b:Person) RETURN b

Generate the Cypher query for the following question.

Question: {question}
Cypher query:
"""

cypher_prompt = PromptTemplate(input_variables=["schema", "question"], template=cypher_template)

qa_template = """
You are an expert at analyzing information from knowledge graphs. Your task is to generate precise answers using only the provided context information.

Below are examples of questions, their corresponding context, and appropriate answers. Use them to help formulate your response.

Example 1:
Question: Who did Cl meet?
Context: [{{'b': {{'id': 'Cm'}}}}]
Answer: Cl met with Cm.

Answer the following question using the provided context.

Question: {question}
Context: {context}
Answer:
"""

qa_prompt = PromptTemplate(input_variables=["question", "context"], template=qa_template)

chain = MemgraphQAChain.from_llm(
    llm=llm,
    graph=graph,
    allow_dangerous_requests=True,
    verbose=True,
    return_intermediate_steps=True,
    cypher_prompt = cypher_prompt,
    qa_prompt = qa_prompt
)

response = chain.invoke("What programs do people participate in, and who participates in what?")
print(f"Intermediate steps: {response['intermediate_steps']}")
print(f"Final response: {response['result']}")
