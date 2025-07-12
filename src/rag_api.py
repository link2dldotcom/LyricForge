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
    mood: str
    style: str
    language: str
    idea: str

# 新增SUNO知识请求模型
class SunoKnowledgeRequest(BaseModel):
    query: str

# 初始化向量数据库
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
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
def load_vector_store(path: str, source_name: str, k: int = 3) -> SourceAddingRetriever:
    vectorstore = FAISS.load_local(
        path, 
        embedding_model, 
        allow_dangerous_deserialization=True
    )
    return SourceAddingRetriever(
        retriever=vectorstore.as_retriever(search_kwargs={"k": k}),  # 使用k参数
        source=source_name
    )

# 初始化检索器
retriever_lyrics = load_vector_store("./faiss_index", "歌词库")
retriever_knowledge = load_vector_store("./faiss_index_knowledge", "知识库")
retriever_suno = load_vector_store("./faiss_index_knowledge", "SUNO知识库", k=15)  # 增加k值以返回更多文档

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

# 新增SUNO知识检索函数
def retrieve_suno(query: str) -> str:
    """从SUNO知识库中检索相关信息"""
    docs = retriever_suno.get_relevant_documents(query)
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
    ),
    # 新增SUNO知识检索工具
    Tool(
        name="SUNOKnowledgeRetriever",
        func=retrieve_suno,
        description="当需要学习SUNO AI使用方法、参数设置、提示词生成技巧或音乐风格配置时使用此工具"
    )
]

# 创建Agent提示模板 - 修复工具调用格式问题
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", """
你是一位专业的音乐创作助手，负责帮助用户生成SUNO AI的歌曲提示词。
请严格按照以下规则操作：

1. 最多思考两次（Thought/Action步骤最多两次）
2. 在第二次思考后必须生成最终答案
3. 如果第一次思考后已经足够，可以直接生成最终答案

重要提示:
1. 提示词必须使用英文
2. 包含情绪、节奏、风格、歌词主题和伴奏描述
3. 格式示例: "positive, happy, uplifting, energetic, upbeat, driving, paceful, metal but dance-friendly, Upbeat pop song with catchy melodies, Energetic drums, bright synths. Lyrics: [阳光海滩的浪漫邂逅]"

可用工具:
{tools}

可用工具名称列表:
{tool_names}  # 必须包含此行以列出可用工具名称

工具调用格式 - 请严格遵守:
1. 思考(Thought)后必须立即跟随Action和Action Input
2. Action行必须只包含工具名称，不能添加任何符号、括号或解释
   - 正确格式: "Action: MusicKnowledgeRetriever"
   - 错误格式: "Action: [MusicKnowledgeRetriever]"
   - 错误格式: "Action: MusicKnowledgeRetriever, 必须是[...]中的一个"
3. Action Input行必须只包含纯文本查询
   - 正确格式: "Action Input: 2025年流行音乐风格"
   - 错误格式: "Action Input: [MusicKnowledgeRetriever] 2025流行风格"
4. 决定回答时必须使用终止标志:
   Thought: 现在可以回答用户的问题了（必须包含此句作为终止标志）
   Final Answer: [最终英文提示词]

工具名称列表（请原样使用）:
- LyricsRetriever
- MusicKnowledgeRetriever
- SUNOKnowledgeRetriever  # 添加此行

{agent_scratchpad}
    """),
    ("user", "{input}")
])

# 创建React Agent
agent = create_react_agent(llm, tools, agent_prompt)

# 严格限制思考次数为2次
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True, 
    handle_parsing_errors=True, 
    max_iterations=2,  # 严格限制最多2次思考
    early_stopping_method="generate"  # 达到限制时强制生成答案
)

@app.post("/generate-suno-prompt")
async def generate_suno_prompt(request: PromptRequest):
    try:
        # 构建更简洁的用户查询
        user_query = (
            f"创作一首歌曲\n"
            f"情绪: {request.mood}\n"
            f"风格: {request.style}\n"
            f"语言: {request.language}\n"
            f"主题: {request.idea}\n"
            "请生成符合SUNO AI要求的英文提示词（最多两次思考）"
        )
        
        # 使用Agent执行任务
        result = agent_executor.invoke({"input": user_query})
        
        # 提取最终输出
        final_output = result.get("output", "")
        
        # 检查是否达到限制
        if "Agent stopped due to iteration limit" in final_output:
            # 尝试提取最后一次思考内容
            if result.get("intermediate_steps"):
                last_step = result["intermediate_steps"][-1]
                if isinstance(last_step[0], str):
                    last_thought = last_step[0].split("Thought: ")[-1]
                else:
                    last_thought = last_step[0].log.split("Thought: ")[-1]
                return {
                    "input": user_query,
                    "suno_prompt": last_thought,
                    "warning": "达到思考次数限制，使用最后思考内容作为提示"
                }
            else:
                # 直接生成提示词
                direct_prompt = await generate_direct_prompt(request)
                return {
                    "input": user_query,
                    "suno_prompt": direct_prompt,
                    "warning": "达到思考次数限制且无中间步骤，使用直接生成方法"
                }
        
        return {
            "input": user_query,
            "suno_prompt": final_output,
            "agent_steps": result.get("intermediate_steps", [])
        }
    except Exception as e:
        return {"error": str(e)}

async def generate_direct_prompt(request: PromptRequest) -> str:
    """当Agent失败时直接生成提示词的备用方法"""
    try:
        # 直接使用LLM生成提示词
        prompt_template = f"""
        用户请求: 创作一首{request.mood}情绪的{request.style}风格歌曲
        语言: {request.language}
        主题: {request.idea}
        
        请生成符合SUNO AI要求的英文提示词，包含:
        - 情绪 (Mood)
        - 节奏 (Tempo)
        - 风格 (Genre/Style)
        - 歌词主题 (Lyric Theme)
        - 伴奏描述 (Instrumentation)
        
        示例格式: 
        "Upbeat pop song with catchy melodies, about summer love. Energetic drums, bright synths. Lyrics: [阳光海滩的浪漫邂逅]"
        """
        
        response = await llm.ainvoke(prompt_template)
        return response.content
    except Exception as e:
        return f"Error: {str(e)}"

# 添加首页路由
@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
@app.get("/suno-knowledge")
async def suno_knowledge_page():
    return templates.TemplateResponse("suno_knowledge.html", {"request": Request(scope={"type": "http"})})
# 新增SUNO知识学习端点
@app.post("/learn-suno-knowledge")
async def learn_suno_knowledge(request: SunoKnowledgeRequest):
    try:
        # 直接检索SUNO知识
        knowledge = retrieve_suno(request.query)
        return {
            "query": request.query,
            "knowledge": knowledge
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    try:
        import uvicorn
        uvicorn.run("rag_api:app", host="0.0.0.0", port=8000, reload=True)
    except Exception as e:
        print(f"程序启动失败: {str(e)}")
        import traceback
        traceback.print_exc()