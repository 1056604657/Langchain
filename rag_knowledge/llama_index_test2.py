import openai
import qdrant_client
from IPython.display import Markdown, display
from llama_index.core.data_structs import IndexDict
from llama_index.core import StorageContext, load_index_from_storage, VectorStoreIndex
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

client = qdrant_client.QdrantClient(url="https://e321d9e8-1d96-4d5a-b985-10a56bd5b28e.us-east4-0.gcp.cloud.qdrant.io:6333",
            api_key="hL3wA6jIDZcP-M65ZhsRUh5ukV5XxJI8nkXaUHCS7sa-7N3Pzv5oKg")
from llama_index.core.schema import TextNode

nodes = [
    TextNode(
        text="りんごとは",
        metadata={"author": "Tanaka", "fruit": "apple", "city": "Tokyo"},
    ),
    TextNode(
        text="Was ist Apfel?",
        metadata={"author": "David", "fruit": "apple", "city": "Berlin"},
    ),
    TextNode(
        text="Orange like the sun",
        metadata={"author": "Jane", "fruit": "orange", "city": "Hong Kong"},
    ),
    TextNode(
        text="Grape is...",
        metadata={"author": "Jane", "fruit": "grape", "city": "Hong Kong"},
    ),
    TextNode(
        text="T-dot > G-dot",
        metadata={"author": "George", "fruit": "grape", "city": "Toronto"},
    ),
    TextNode(
        text="6ix Watermelons",
        metadata={
            "author": "George",
            "fruit": "watermelon",
            "city": "Toronto",
        },
    ),
]

openai.api_key = "sk-A2dkpolqc9XtWx6xC36e2498A24748489d3348Ef1a23395a"
openai.base_url = "https://api.xty.app/v1"
vector_store = QdrantVectorStore(
    client=client, collection_name="fruit_collection"
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)


VectorStoreIndex(nodes, storage_context=storage_context)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store, storage_context=storage_context)
# Use filters directly from qdrant_client python library
# View python examples here for more info https://qdrant.tech/documentation/concepts/filtering/

filters = Filter(
    should=[
        Filter(
            must=[
                FieldCondition(
                    key="fruit",
                    match=MatchValue(value="apple"),
                ),
                FieldCondition(
                    key="city",
                    match=MatchValue(value="Tokyo"),
                ),
            ]
        ),
        Filter(
            must=[
                FieldCondition(
                    key="fruit",
                    match=MatchValue(value="grape"),
                ),
                FieldCondition(
                    key="city",
                    match=MatchValue(value="Toronto"),
                ),
            ]
        ),
    ]
)

retriever = index.as_retriever(vector_store_kwargs={"qdrant_filters": filters})

response = retriever.retrieve("Who makes grapes?")
for node in response:
    print("node", node.score)
    print("node", node.text)
    print("node", node.metadata)