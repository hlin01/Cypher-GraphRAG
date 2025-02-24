import os
from dotenv import load_dotenv

from langchain_community.graphs import MemgraphGraph
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI

load_dotenv()
os.environ["OPENAI_API_KEY"]

url = os.environ.get("MEMGRAPH_URI", "bolt://localhost:7687")
username = os.environ.get("MEMGRAPH_USERNAME", "")
password = os.environ.get("MEMGRAPH_PASSWORD", "")

# initialize memgraph connection
graph = MemgraphGraph(
    url=url, username=username, password=password, refresh_schema=False
)

# read txt file
with open("anonymized_data_cl1.txt", "r") as f:
    notes = f.readlines()

llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
llm_transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=[
        "Case",
        "Person",
        "Organization",
        "Location",
        "Document",
        "Communication",
        "Event",
        "Order",
        "Program",
        "Concept",
        "Date",
        "Time",
        "Financial",
        "Property",
        "Account",
        "System",
        "Activity",
        "ContactInfo",
    ],
    allowed_relationships=[
        "ASSOCIATED_WITH",
        "BELONGS_TO",
        "LOCATED_AT",
        "REFERENCES",
        "CREATED_BY",
        "PARTICIPATES_IN",
        "CONNECTED_TO",
        "LINKED_TO",
        "HAS_STATUS",
        "INCLUDES",
        "OCCURRED_ON",
        "HAS_CONTACT",
        "USES",
        "ASSIGNED_TO",
        "EXCHANGES_WITH",
        "OWNS",
        "MADE_BY",
        "AFFECTS",
    ],
    node_properties=False,
    relationship_properties=False
)

documents = [Document(page_content=note.strip()) for note in notes if note.strip()]
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
