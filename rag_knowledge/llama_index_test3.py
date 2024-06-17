import os
from llama_index.core import Settings
import numpy as np
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
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
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.question_gen.types import SubQuestion
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
# iterate through sub_question items captured in SUB_QUESTION event
from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.embeddings import resolve_embed_model

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

Settings.llm = Ollama(model="wangshenzhi/llama3-8b-chinese-chat-ollama-q8:latest", base_url="http://10.10.10.15:11434",
                      requst_timeout=600)
#Settings.llm = Ollama(model="gpt-3.5-turbo", api_key="sk-Zekmsmp9EOEL9IfX432624CfCb0144AbB946875e9d15BeA8", api_base='https://api.xty.app/v1')
#Settings.embed_model = OpenAIEmbedding(api_key="sk-Zekmsmp9EOEL9IfX432624CfCb0144AbB946875e9d15BeA8", api_base='https://api.xty.app/v1')
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Using the LlamaDebugHandler to print the trace of the sub questions
# captured by the SUB_QUESTION callback event type
llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])

Settings.callback_manager = callback_manager

# load data
pg_essay = SimpleDirectoryReader(input_dir="./data/xigua/").load_data()

# build index and query engine
vector_query_engine = VectorStoreIndex.from_documents(
    pg_essay,
    use_async=True,
).as_query_engine()

# setup base query engine as tool
query_engine_tools = [
    QueryEngineTool(
        query_engine=vector_query_engine,
        metadata=ToolMetadata(
            name="西瓜",
            description="西瓜与番茄的介绍"
        )
    )
]

query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=query_engine_tools,
    use_async=True
)

response = query_engine.query(
    "番茄的形态特征的生长环境是什么?"
)
print(response)

for i, (start_event, end_event) in enumerate(
        llama_debug.get_event_pairs(CBEventType.SUB_QUESTION)
):
    qa_pair = end_event.payload[EventPayload.SUB_QUESTION]
    print("Sub Question " + str(i) + ": " + qa_pair.sub_q.sub_question.strip())
    print("Answer: " + qa_pair.answer.strip())
    print("====================================")
