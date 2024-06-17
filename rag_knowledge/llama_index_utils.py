import os

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core import Settings
from llama_index.core.agent import FunctionCallingAgentWorker
# iterate through sub_question items captured in SUB_QUESTION event
from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.query_engine.retriever_query_engine import (
    RetrieverQueryEngine,
)
from llama_index.retrievers.bm25.base import BM25Retriever

# Using the LlamaDebugHandler to print the trace of the sub questions
# captured by the SUB_QUESTION callback event type
llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])

Settings.callback_manager = callback_manager

for i, (start_event, end_event) in enumerate(
        llama_debug.get_event_pairs(CBEventType.SUB_QUESTION)
):
    qa_pair = end_event.payload[EventPayload.SUB_QUESTION]
    print("Sub Question " + str(i) + ": " + qa_pair.sub_q.sub_question.strip())
    print("Answer: " + qa_pair.answer.strip())
    print("====================================")


def get_openai_llm():
    return OpenAI(api_key=os.getenv('OPENAI_API_KEY'), api_base=os.getenv('OPENAI_API_BASE'))


def get_openai_embedding():
    return OpenAIEmbedding(api_key=os.getenv('OPENAI_API_KEY'), api_base=os.getenv('OPENAI_API_BASE'))


def build_document(docs, filename, filepath):
    return [Document(text=t, extra_info={'file_name': filename, 'file_path': filepath}) for t in docs]


def qdrant_add_points(client, collection_name, embed_model, documents):
    vector_store = QdrantVectorStore(
        client=client, collection_name=collection_name
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex(nodes=documents, storage_context=storage_context, embed_model=embed_model)
    return True


def qdrant_index(client, collection_name):
    qdrant_vector_store = QdrantVectorStore(
        client=client, collection_name=collection_name
    )
    storage_context = StorageContext.from_defaults(vector_store=qdrant_vector_store)
    index = VectorStoreIndex.from_vector_store(
        vector_store=qdrant_vector_store, storage_context=storage_context
    )
    return index


def get_query_engine_tools(engine_tools_last_params):
    query_engine_tools = []
    for engine_tools_last_param in engine_tools_last_params:
        query_engine_tool = QueryEngineTool(query_engine=engine_tools_last_param[0],
                                            metadata=ToolMetadata(
                                                name=engine_tools_last_param[1],
                                                description=engine_tools_last_param[2]
                                            ))
        query_engine_tools.append(query_engine_tool)
    return query_engine_tools


def get_agent(query_engine_tools):
    agent_worker = FunctionCallingAgentWorker.from_tools(
        query_engine_tools,
        verbose=True,
        allow_parallel_tool_calls=False,
        system_prompt="""\
            你是一个回答问题的专家.你必须使用其中一个提供的工具来回答问题，如果工具不能获取，则回答不知道
        """
    )
    agent = agent_worker.as_agent()
    return agent


def get_sub_query_engine(query_engine_tools, llm):
    return SubQuestionQueryEngine.from_defaults(
        query_engine_tools
    )

