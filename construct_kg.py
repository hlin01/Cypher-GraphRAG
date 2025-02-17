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
llm_transformer = LLMGraphTransformer(llm=llm)
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

cypher_template = """
Your task is to directly translate natural language inquiry into precise and executable Cypher query for Memgraph database.
You will utilize a provided database schema to understand the structure, nodes and relationships within the Memgraph database.

Instructions:

- Respond with ONE and ONLY ONE query.

- Use provided node and relationship labels and property names from the schema which describes the database's structure. Upon receiving a user question, synthesize the schema to craft a precise Cypher query that directly corresponds to the user's intent.

- Generate valid executable Cypher queries on top of Memgraph database. Any explanation, context, or additional information that is not a part of the Cypher query syntax should be omitted entirely.

- Use Memgraph MAGE procedures instead of Neo4j APOC procedures.

- Do not include any explanations or apologies in your responses. Only answer the question asked.

- Do not include additional questions. Only the original user question.

- Do not include any text except the generated Cypher statement.

- For queries that ask for information or functionalities outside the direct generation of Cypher queries, use the Cypher query format to communicate limitations or capabilities. For example: RETURN "I am designed to generate Cypher queries based on the provided schema only."

Here is the schema information:
{schema}

With all the above information and instructions, generate Cypher query for the user question.

The question is:
{question}
"""

cypher_prompt = PromptTemplate(input_variables=["schema", "question"], template=cypher_template)

chain = MemgraphQAChain.from_llm(
    ChatOpenAI(temperature=0),
    graph=graph,
    model_name="gpt-4o",
    allow_dangerous_requests=True,
    verbose=False,
    return_intermediate_steps=True,
    cypher_prompt = cypher_prompt
)

response = chain.invoke("Where did CM arrive at?")
print(f"Intermediate steps: {response['intermediate_steps']}")
print(f"Final response: {response['result']}")
