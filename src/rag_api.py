from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
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
from pathlib import Path
from typing import List, Tuple, Any, Dict
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate

# 设置Hugging Face国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 初始化 FastAPI
app = FastAPI()

# 挂载静态文件
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")
# 初始化模板引擎
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")
# 跨域配置
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

# 初始化向量数据库
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddingsings': True}
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

# 加载向量库
def load_vector_store(path: str, source_name: str) -> SourceAddingRetriever:
    vectorstore = FAISS.load_local(
        path, 
        embedding_model, 
        allow_dangerous_deserialization=True
    )
    return SourceAddingRetriever(
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        source=source_name
    )

# 初始化检索器
retriever_lyrics = load_vector_store("./faiss_index", "歌词库")
retriever_knowledge = load_vector_store("./faiss_index_knowledge", "知识库")

# 初始化本地 LLM
llm = ChatOllama(model="deepseek-r1:7b", base_url="http://localhost:11434", temperature=0.2)

# 定义检索工具函数
def retrieve_lyrics(query: str) -> str:
    """从歌词库中检索相关信息"""
    docs = retriever_lyrics.get_relevant_documents(query)
    return format_documents(docs)

def retrieve_knowledge(query: str) -> str:
    """从音乐知识库中检索相关信息"""
    docs = retriever_knowledge.get_relevant_documents(query)
    return format_documents(docs)

def format_documents(docs: List[Document]) -> str:
    """格式化检索结果"""
    return "\n\n".join([
        f"来源: {doc.metadata.get('source', '未知')}\n内容: {doc.page_content}" 
        for doc in docs
    ])

# 创建工具列表
tools = [
    Tool(
        name="LyricsRetriever",
        func=retrieve_lyrics,
        description="当需要查找歌曲歌词、歌词风格或歌词结构时使用此工具"
    ),
    Tool(
        name="MusicKnowledgeRetriever",
        func=retrieve_knowledge,
        description="当需要音乐理论、歌曲创作技巧或音乐风格信息时使用此工具"
    )
]

# 创建Agent提示模板
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    你是一位专业的音乐创作助手，负责帮助用户生成SUNO AI的歌曲提示词。
    请根据用户请求，智能选择使用提供的工具获取相关信息，然后生成高质量的歌曲提示词。
     
     重要提示:
     1. 提示词必须使用英文
     2. 包含情绪、节奏、风格、歌词主题和伴奏描述
     3. 格式示例: "Upbeat pop song with catchy melodies, about summer love. Energetic drums, bright synths. Lyrics: [阳光海滩的浪漫邂逅]"
     
     工具说明:
     {tools}
     
     使用工具时请遵循以下格式:
     Question: 用户的问题
     Thought: 需要思考是否需要使用工具
     Action: 工具名称，必须是[{tool_names}]中的一个
     Action Input: 工具的输入参数
     Observation: 工具返回的结果
     ...(可以重复多次Thought/Action/Action Input/Observation)
     Thought: 现在可以回答用户的问题了
     Final Answer: 最终答案
     
     开始思考:
     {agent_scratchpad}
    """),
    ("user", "{input}")
])

# 创建React Agent
agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

@app.post("/generate-suno-prompt")
async def generate_prompt(request: PromptRequest):
    try:
        user_query = (
            f"用户请求: 创作一首{request.mood}情绪的{request.style}风格歌曲\n"
            f"语言: {request.language}\n"
            f"主题: {request.idea}\n"
            "请生成符合SUNO AI要求的英文提示词"
        )
        
        # 使用Agent执行任务
        result = agent_executor.invoke({"input": user_query})
        
        return {
            "input": user_query,
            "suno_prompt": result["output"],
            "agent_steps": result.get("intermediate_steps", [])
        }
    except Exception as e:
        return {"error": str(e)}

# 添加首页路由
@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    try:
        import uvicorn
        uvicorn.run("rag_api:app", host="0.0.0.0", port=8000, reload=True)
    except Exception as e:
        print(f"程序启动失败: {str(e)}")
        import traceback
        traceback.print_exc()