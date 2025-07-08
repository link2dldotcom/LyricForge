from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import shutil
import os
from langchain_core.documents import Document
import uuid

# 确保使用与创建索引时相同的嵌入模型
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 定义索引路径
MAIN_INDEX_PATH = "../faiss_index"
KNOWLEDGE_INDEX_PATH = "../faiss_index_knowledge"
BACKUP_INDEX_PATH = "../faiss_index_backup"

# 备份主索引（重要！）
if not os.path.exists(BACKUP_INDEX_PATH):
    shutil.copytree(MAIN_INDEX_PATH, BACKUP_INDEX_PATH)
    print(f"主索引已备份至: {BACKUP_INDEX_PATH}")

# 加载两个索引
print("加载索引中...")
main_db = FAISS.load_local(MAIN_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
knowledge_db = FAISS.load_local(KNOWLEDGE_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

# 检查向量维度是否匹配
main_dim = main_db.index.d
knowledge_dim = knowledge_db.index.d
print(f"主索引维度: {main_dim}, 知识索引维度: {knowledge_dim}")

if main_dim != knowledge_dim:
    print(f"维度不匹配，将使用主索引维度({main_dim})重新嵌入知识文档...")
    # 提取知识索引文档内容
    knowledge_texts = [doc.page_content for doc in knowledge_db.docstore._dict.values()]
    knowledge_metadatas = [doc.metadata for doc in knowledge_db.docstore._dict.values()]
    
    # 使用主索引的嵌入模型重新生成向量
    knowledge_embeddings = embeddings.embed_documents(knowledge_texts)
    
    # 创建新的知识索引（使用主索引维度）
    knowledge_db = FAISS.from_embeddings(
        list(zip(knowledge_texts, knowledge_embeddings)),
        embeddings,
        metadatas=knowledge_metadatas
    )

# 获取知识索引中的所有文档并添加ID
knowledge_docs = []
for doc in knowledge_db.docstore._dict.values():
    doc_id = doc.id if hasattr(doc, 'id') and doc.id else str(uuid.uuid4())
    new_doc = Document(page_content=doc.page_content, metadata=doc.metadata, id=doc_id)
    knowledge_docs.append(new_doc)

print(f"发现{len(knowledge_docs)}个知识文档，准备合并...")

# 为知识文档添加来源标记（可选）
for doc in knowledge_docs:
    doc.metadata["source"] = doc.metadata.get("source", "suno_knowledge")

# 合并索引
main_db.add_documents(knowledge_docs)

# 保存合并后的索引
main_db.save_local(MAIN_INDEX_PATH)
print(f"索引合并完成！合并后总文档数: {main_db.index.ntotal}")
