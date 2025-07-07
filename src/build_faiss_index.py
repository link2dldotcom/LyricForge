# build_faiss_index.py

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import os

# 示例歌词文档（你可以替换为自己的）
docs = [
    Document(page_content="阳光洒满你的笑脸，我的心跳随你蔓延...", metadata={"title": "夏日恋歌", "style": "流行"}),
    Document(page_content="我在北漂的地铁上思考自由的代价...", metadata={"title": "城市孤影", "style": "说唱"}),
    Document(page_content="一把吉他唱尽青春的年华...", metadata={"title": "青春岁月", "style": "民谣"}),
]

# 嵌入模型（建议使用中文专用）
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

# 构建索引
db = FAISS.from_documents(docs, embedding_model)

# 保存索引到目录
os.makedirs("faiss_index", exist_ok=True)
db.save_local("faiss_index")

print("✅ FAISS 索引已保存到 faiss_index/")
