# build_faiss_index.py

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import os, json

# 歌词数据源路径（JSONL 格式，每行一个 JSON）
SOURCE_FILE = "lyrics_dataset.jsonl"

# 读取语料库文件
lyrics_docs = []
with open(SOURCE_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        content = data.get("lyrics", "")
        metadata = {
            "title": data.get("title", ""),
            "mood": data.get("mood", ""),
            "style": data.get("style", ""),
            "language": data.get("language", ""),
            "suno_prompt": data.get("suno_prompt", "")
        }
        lyrics_docs.append(Document(page_content=content, metadata=metadata))

# 嵌入模型
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

# 构建 FAISS 索引
vectorstore = FAISS.from_documents(lyrics_docs, embedding_model)

# 保存索引
os.makedirs("faiss_index", exist_ok=True)
vectorstore.save_local("faiss_index")

print("✅ 已从 lyrics_dataset.jsonl 构建并保存向量索引")
