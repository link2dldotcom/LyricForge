# build_faiss_index.py

from langchain.vectorstores import FAISS

# 修改为
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
import os, json
from PyPDF2 import PdfReader  # 新增：PDF处理依赖

# 设置Hugging Face国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 配置：输入目录和支持的文件类型
INPUT_DIRECTORY = "lyricsSource"  # 存放待导入文件的目录
SUPPORTED_EXTENSIONS = {'.txt', '.json', '.jsonl', '.pdf'}

# -------------------------- 文件加载器 --------------------------

def load_txt(file_path):
    """加载TXT文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return [Document(
        page_content=content,
        metadata={"source": file_path, "type": "txt"}
    )]


def load_json(file_path):
    """加载JSON文件（支持单对象或对象列表）"""
    docs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        items = data if isinstance(data, list) else [data]
        
        for item in items:
            content = item.get("content", item.get("lyrics", str(item)))
            metadata = {"source": file_path, "type": "json"}
            metadata.update({k: v for k, v in item.items() if k != "content"})
            docs.append(Document(page_content=content, metadata=metadata))
    return docs


def load_jsonl(file_path):
    """加载JSONL文件（每行一个JSON对象）"""
    docs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                content = data.get("lyrics", data.get("content", str(data)))
                metadata = {
                    "source": file_path,
                    "type": "jsonl",
                    "title": data.get("title", ""),
                    "mood": data.get("mood", ""),
                    "style": data.get("style", "")
                }
                docs.append(Document(page_content=content, metadata=metadata))
    return docs


def load_pdf(file_path):
    """加载PDF文件（提取每页文本）"""
    docs = []
    reader = PdfReader(file_path)
    for page_num, page in enumerate(reader.pages, 1):
        content = page.extract_text()
        if content.strip():
            docs.append(Document(
                page_content=content,
                metadata={
                    "source": file_path,
                    "type": "pdf",
                    "page": page_num,
                    "total_pages": len(reader.pages)
                }
            ))
    return docs

# -------------------------- 主流程 --------------------------

# 加载器映射表
LOADERS = {
    '.txt': load_txt,
    '.json': load_json,
    '.jsonl': load_jsonl,
    '.pdf': load_pdf
}

# 批量加载所有文件
all_docs = []
for root, _, files in os.walk(INPUT_DIRECTORY):
    for file in files:
        ext = os.path.splitext(file)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            file_path = os.path.join(root, file)
            try:
                docs = LOADERS[ext](file_path)
                all_docs.extend(docs)
                print(f"✅ 加载成功: {file_path} (文档数: {len(docs)})")
            except Exception as e:
                print(f"❌ 加载失败: {file_path} (错误: {str(e)})")

if not all_docs:
    print("⚠️ 未找到任何文档，请检查 INPUT_DIRECTORY 配置")
    exit(1)

# 构建向量索引
print(f"\n开始构建向量索引（总文档数: {len(all_docs)}）...")
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vectorstore = FAISS.from_documents(all_docs, embedding_model)

# 保存索引
output_dir = "faiss_index"
os.makedirs(output_dir, exist_ok=True)
vectorstore.save_local(output_dir)
print(f"\n✅ 索引构建完成，已保存至 {output_dir}（总向量数: {len(vectorstore.docstore._dict)}）")
