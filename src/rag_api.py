from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates  # 添加这行
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
import os
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain.retrievers import EnsembleRetriever
from pathlib import Path  # 添加此行导入

# 设置Hugging Face国内镜像（如果需要）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# print("当前HF_ENDPOINT配置:", os.getenv("HF_ENDPOINT"))

# 初始化 FastAPI
app = FastAPI()

# 挂载静态文件
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")
# 初始化模板引擎 - 修改这行
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")
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
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

faiss_index_path = "./faiss_index"
vectorstore = FAISS.load_local(faiss_index_path, embedding_model, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

# 初始化本地 LLM（使用 deepseek-r1:7b）
llm = ChatOllama(model="deepseek-r1:7b", base_url="http://localhost:11434")

# 创建QA链
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# 创建带源信息的检索器
class SourceAddingRetriever(BaseRetriever):
    retriever: BaseRetriever = Field(..., description="基础检索器")
    source: str = Field(..., description="来源标识")

    def __init__(self, retriever: BaseRetriever, source: str):
        super().__init__(retriever=retriever, source=source)

    def _get_relevant_documents(self, query: str) -> list[Document]:
        documents = self.retriever.get_relevant_documents(query)
        for doc in documents:
            doc.metadata["source"] = self.source
        return documents

# 加载歌词向量库
vectorstore_lyrics = FAISS.load_local(
    "./faiss_index", 
    embedding_model, 
    allow_dangerous_deserialization=True
)

# 加载知识向量库
vectorstore_knowledge = FAISS.load_local(
    "./faiss_index_knowledge", 
    embedding_model, 
    allow_dangerous_deserialization=True
)

# 创建两个源检索器
retriever_lyrics = SourceAddingRetriever(retriever=vectorstore_lyrics.as_retriever(), source="lyrics")
retriever_knowledge = SourceAddingRetriever(retriever=vectorstore_knowledge.as_retriever(), source="knowledge")

# 合并检索器
ensemble_retriever = EnsembleRetriever(
    retrievers=[retriever_lyrics, retriever_knowledge],
    weights=[0.5, 0.5]
)

# 创建QA链时使用合并后的检索器
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=ensemble_retriever,
    return_source_documents=True
)

@app.post("/generate-suno-prompt")
async def generate_prompt(request: PromptRequest):
    try:
        user_query = f"我想写一首{request.mood}的{request.style}风格歌曲，语言是{request.language}，主题是：{request.idea}。请生成一段英文的 SUNO 提示词，用于歌曲生成，包括情绪、节奏、风格、歌词、伴奏。"
        result = qa_chain.invoke({"query": user_query})  # 使用 invoke 方法
        return {
            "input": user_query,
            "suno_prompt": result["result"]  # 返回 result 键的内容
        }
    except Exception as e:
        return {"error": str(e)}

# 添加首页路由
@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
    return {"prompt": generated_prompt}  # 确保返回格式为 {"prompt": "生成的提示词"}

if __name__ == "__main__":
    try:
        import uvicorn
        uvicorn.run("rag_api:app", host="0.0.0.0", port=8000, reload=True)
    except Exception as e:
        print(f"程序启动失败: {str(e)}")
        import traceback
        traceback.print_exc()