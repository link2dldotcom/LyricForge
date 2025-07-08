from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain.chains import RetrievalQA
# 正确的FAISS导入方式
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama  # 新增此行
import os
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.retrievers import EnsembleRetriever
from pydantic import Field  # 新增此行导入Field

# 设置Hugging Face国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
print("当前HF_ENDPOINT配置:", os.getenv("HF_ENDPOINT"))

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
# 替换旧导入
from langchain_huggingface import HuggingFaceEmbeddings  # 新导入路径

# 确保已安装依赖
# pip install langchain-huggingface
from langchain_huggingface import HuggingFaceEmbeddings

# 初始化嵌入模型（放在加载FAISS索引之前）
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# 确保此初始化代码位于加载FAISS索引之前
faiss_index_path = "./faiss_index"
vectorstore = FAISS.load_local(faiss_index_path, embedding_model, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

# 初始化本地 LLM（使用 deepseek-r1:7b）
# LLM初始化代码保持不变
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
    retriever: BaseRetriever = Field(..., description="基础检索器")  # 现在Field已定义
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
        user_query = f"我想写一首{request.mood}的{request.style}风格歌曲，语言是{request.language}，主题是：{request.idea}。请生成一段英文的 SUNO 提示词，用于歌曲生成，包括情绪、节奏、风格、性别、伴奏。"
        result = qa_chain.run(user_query)
        return {
            "input": user_query,
            "suno_prompt": result
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
async def root():
    return {"message": "LyricForge RAG API running."}


if __name__ == "__main__":
    try:
        # 现有启动代码
        import uvicorn
        uvicorn.run("rag_api:app", host="0.0.0.0", port=8000, reload=True)
    except Exception as e:
        print(f"程序启动失败: {str(e)}")
        import traceback
        traceback.print_exc()