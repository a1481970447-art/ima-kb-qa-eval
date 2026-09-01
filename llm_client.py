"""混元 TokenHub LLM 客户端（OpenAI 兼容协议）。

D3 的 RAG pipeline 与 D4 的评分器都从这里拿模型调用能力：
    from llm_client import chat, embed

直接运行本文件可做冒烟测试：
    python llm_client.py
"""
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # 读取项目根目录 .env

BASE_URL = os.getenv("LLM_BASE_URL", "https://tokenhub.tencentmaas.com/v1")
API_KEY = os.environ["HUNYUAN_API_KEY"]
CHAT_MODEL = os.getenv("LLM_MODEL", "hy3")
EMBED_MODEL = os.getenv("EMBED_MODEL", "kinfra-text-embedding-0.6b")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def chat(
    question: str,
    system: str = "",
    model: str | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.3,
) -> str:
    """单轮问答。

    注意：hy3 是思考模型，回答前会先消耗 reasoning_content，
    max_tokens 给太小会得到空回复（finish_reason=length）。
    """
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": question}
    ]
    resp = client.chat.completions.create(
        model=model or CHAT_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""


def embed(texts: list[str], model: str | None = None) -> list[list[float]]:
    """批量文本向量化（D3 检索层用）。

    TokenHub 的 /embeddings 仅支持 encoding_format="float"
    （openai 新版 SDK 默认发 base64 会报 400001）。
    """
    resp = client.embeddings.create(
        model=model or EMBED_MODEL, input=texts, encoding_format="float"
    )
    return [d.embedding for d in resp.data]


if __name__ == "__main__":
    print(f"chat 模型: {CHAT_MODEL}  |  embed 模型: {EMBED_MODEL}")
    print("-" * 40)
    print("问答冒烟测试：", chat("用一句话介绍 ima 是什么产品")[:200])
    print("-" * 40)
    vec = embed(["知识库问答质量评测"])
    print(f"embedding 冒烟测试：维度 = {len(vec[0])}")
