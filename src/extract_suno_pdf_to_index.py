# extract_suno_pdf_to_index.py

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from pdfminer.high_level import extract_text
import os

# 输入 PDF 路径
pdf_path = "src/sunoKnowledge2.pdf"  # 替换为你的 PDF 文件路径

# 1. 提取 PDF 文本
print("📖 正在提取 PDF 文本...")
full_text = extract_text(pdf_path)

# 2. 分段处理（以换行或空行为段落）
paragraphs = [p.strip() for p in full_text.split("\n\n") if len(p.strip()) > 40]
print(f"✅ 提取段落数：{len(paragraphs)}")

# 3. 转换为 Document 列表
docs = [Document(page_content=p, metadata={"source": "suno_pdf"}) for p in paragraphs]

# 4. 向量嵌入模型
print("🔍 正在加载嵌入模型...")
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

# 5. 构建并保存 FAISS 向量库
print("🧠 正在构建知识向量索引...")
vectorstore = FAISS.from_documents(docs, embedding_model)
os.makedirs("faiss_index_knowledge", exist_ok=True)
vectorstore.save_local("faiss_index_knowledge")

print("🎉 知识库构建完成，已保存至 faiss_index_knowledge/")
