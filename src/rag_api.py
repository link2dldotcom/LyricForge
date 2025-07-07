from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
import os

# 初始化 FastAPI
app = FastAPI()

# 跨域配置（可根据需要调整）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 请求体结构
class PromptRequest(BaseModel):
    idea: str
    language: str = "zh"
    mood: str = ""
    style: str = ""

# 初始化向量数据库（加载本地 FAISS 文件）
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
faiss_index_path = "./faiss_index"
vectorstore = FAISS.load_local(faiss_index_path, embedding_model, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

# 初始化本地 LLM（使用 deepseek-r1:7b）
llm = Ollama(model="deepseek-r1:7b", base_url="http://localhost:11434")

# 构建 RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

@app.post("/generate-suno-prompt")
async def generate_prompt(request: PromptRequest):
    user_query = f"我想写一首{request.mood}的{request.style}风格歌曲，语言是{request.language}，主题是：{request.idea}。请生成一段英文的 SUNO 提示词，用于歌曲生成，包括情绪、节奏、风格、性别、伴奏。"
    result = qa_chain.run(user_query)
    return {
        "input": user_query,
        "suno_prompt": result
    }

@app.get("/")
async def root():
    return {"message": "LyricForge RAG API running."}
