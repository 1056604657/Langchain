import os

from llama_index.core.postprocessor import SentenceTransformerRerank

os.environ["OPENAI_API_BASE"] = 'https://api.xty.app/v1'
os.environ["OPENAI_API_KEY"] = 'sk-4diRZahfSd7bSZ1S4819A936249645389dBbD5Bb4e85E37f'

MODELS = [
    "llama3-8b-chines",
    "gpt-3.5-turbo-1106", # 最新的 GPT-3.5 Turbo 模型，具有改进的指令遵循、JSON 模式、可重现输出、并行函数调用等。最多返回 4,096 个输出标记。
    "gpt-3.5-turbo",  # 当前指向 gpt-3.5-turbo-0613 。自 2023 年 12 月 11  日开始指向gpt-3.5-turbo-1106。
    "gpt-3.5-turbo-16k",  # 当前指向 gpt-3.5-turbo-0613 。将指向gpt-3.5-turbo-1106 2023 年 12 月 11 日开始。
]
#DEFAULT_MODEL = MODELS[-2]
DEFAULT_MODEL = MODELS[0]

MODEL_TO_MAX_TOKENS = {
    "llama3-8b-chines": 4096,
    "gpt-3.5-turbo-1106": 4096,
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-16k": 16385,
}

# Define all embeddings and rerankers
RERANKERS = {
    "WithoutReranker": "None",
    "bge-reranker-base": SentenceTransformerRerank(model="BAAI/bge-reranker-base", top_n=2),
    "bge-reranker-large": SentenceTransformerRerank(model="BAAI/bge-reranker-large", top_n=2)
}

EMBED_MODEL = "text-embedding-ada-002"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

DEFAULT_MAX_TOKENS = 4000