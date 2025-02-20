import os
from dotenv import load_dotenv

from langchain_community.chains.graph_qa.memgraph import MemgraphQAChain
from langchain_community.graphs import MemgraphGraph
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_openai import ChatOpenAI

from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer

load_dotenv()
os.environ["OPENAI_API_KEY"]

url = os.environ.get("MEMGRAPH_URI", "bolt://localhost:7687")
username = os.environ.get("MEMGRAPH_USERNAME", "")
password = os.environ.get("MEMGRAPH_PASSWORD", "")

# initialize memgraph connection
graph = MemgraphGraph(
    url=url, username=username, password=password, refresh_schema=False
)

# line 14
text = "CM arrived at CLs home 3pm as scheduled with CL. CL answered the door and met with CM outside. CL has a friend over and CL thought CM would not show up because of how CL communicated with CM yesterday. CM reminded CL that they scheduled this appointment last week and CM just spoke with CL yesterday. CM confirmed with CL that their application fee was paid online for the private landlord duplex in Leander. CL is depressed and broke up with their boyfriend. CL is not feeling hopeful about the future since CL admits to being co-dependent and likes always having a partner. CL has a friend over and does not want to meet with CM today. CLs friend is helping CL not return to substance use and CM suggested CL do something healthy and positive that makes them feel like themself. CL feels safe and supported by their friend. CL would possibly like to meet on Friday but is not sure yet. CL asked CM to call CL on Thursday. CM agreed to call CL Thursday and if CM does not hear from CL on Thursday, they will not meet on Friday. CL understands and agrees. CM and CL tentatively scheduled to meet Friday 8/19/22 at 2pm at CLs home. INCENTIVES GIVEN: 1 GIFT CARD."

llm = ChatOpenAI(temperature=0, model_name="gpt-4o")

llm_transformer = LLMGraphTransformer(
    llm=llm,
    node_properties=False,
    relationship_properties=False
)

documents = [Document(page_content=text)]
graph_documents = llm_transformer.convert_to_graph_documents(documents)

print(graph_documents)

# make sure the database is empty
graph.query("STORAGE MODE IN_MEMORY_ANALYTICAL")
graph.query("DROP GRAPH")
graph.query("STORAGE MODE IN_MEMORY_TRANSACTIONAL")

# create knowledge graph
graph.add_graph_documents(graph_documents)

graph.refresh_schema()
print(graph.get_schema)

# ---------------------------- MOVE FOLLOWING TO ANOTHER FILE ----------------------------

cypher_template = """
You are an expert in crafting precise and executable Cypher queries for a Memgraph database based on a given schema.

Instructions:
- Generate ONE and ONLY ONE valid Cypher query that directly answers the user's natural language question.
- Use only the node labels, relationship types, and property names provided in the schema.
- The query must be executable on a Memgraph database and use Memgraph MAGE procedures when applicable (do not use Neo4j APOC procedures).
- Do not include any text other than the Cypher query (no explanations, context, questions, or apologies).
- Do not include the text 'cypher' before the query.

Schema:
{schema}

Generate the Cypher query for the following question:
{question}
"""

cypher_prompt = PromptTemplate(input_variables=["schema", "question"], template=cypher_template)

qa_template = """
You are an expert in analyzing structured context from a knowledge graph. Using only the context provided below, generate a clear, concise, and factual answer to the following question.

Question:
{question}

Context:
{context}
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

response = chain.invoke("What happened to Cl?")
print(f"Intermediate steps: {response['intermediate_steps']}")
print(f"Final response: {response['result']}")
