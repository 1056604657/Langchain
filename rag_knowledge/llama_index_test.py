from llama_index.core import Settings
import numpy as np
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss
from IPython.display import Markdown, display
from llama_index.core import (
    SimpleDirectoryReader,
    load_index_from_storage,
    VectorStoreIndex,
    StorageContext
)

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

Settings.llm = Ollama(model="llama3", base_url="http://10.10.10.15:11434")

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

documents = SimpleDirectoryReader("./data").load_data()


d = 384
faiss_index = faiss.IndexFlat(d)
vector_store = FaissVectorStore(faiss_index=faiss_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

# save index to disk
index.storage_context.persist(persist_dir="./storage")

# load index from disk
vector_store = FaissVectorStore.from_persist_dir()
storage_context = StorageContext.from_defaults(
    vector_store=vector_store, persist_dir="./storage"
)
index = load_index_from_storage(storage_context=storage_context)


# set Logging to DEBUG for more detailed outputs
query_engine = index.as_query_engine(similarity_top_k=3, response_mode="tree_summarize", verbose=True)
response = query_engine.query("What did the author do growing up, please answer in chinese?")

display(Markdown(f"<b>{response}</b>"))


