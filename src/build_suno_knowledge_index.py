# build_suno_knowledge_index.py

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import os

# 模拟：从 PDF 文档提取的内容（此处用手动整理后的段落替代）
suno_paragraphs = [
    "Dance: Punchy 4/4 beats, electro bass, catchy synths, pop vocals, bright pads, club-ready mixes, energetic drops.",
    "Disco: Groovy disco beats, funky bass, retro synths, soulful vocals, smooth transitions, feel-good vibes.",
    "Techno: Minimal techno rhythms, deep bass, ambient pads, sparse vocals, hypnotic loops, dark club sounds.",
    "Dubstep: Wobbles, heavy drops, gritty basslines, syncopated rhythms, dark atmospheres, intense energy.",
    "Ambient: Ambient soundscapes, atmospheric pads, subtle rhythms, tranquil melodies, introspective mood.",
    "Reggae: Offbeat guitars, warm basslines, laid-back vocals, island vibes, positive energy.",
    "Lyrical: Soulful lyrics, emotional melodies, intricate harmonies, heartfelt delivery.",
    "Jazz: Jazz improvisation, bluesy guitar solos, brass harmonies, smooth melodies, swinging rhythm.",
    "Trance: Uplifting trance arpeggios, euphoric melodies, layered synths, driving beats, wide stereo imaging."
]

# 构建文档结构
knowledge_docs = [Document(page_content=p, metadata={"source": "suno_guide"}) for p in suno_paragraphs]

# 嵌入模型
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

# 构建 FAISS 向量数据库
vectorstore = FAISS.from_documents(knowledge_docs, embedding_model)

# 保存到 faiss_index_knowledge 目录
os.makedirs("faiss_index_knowledge", exist_ok=True)
vectorstore.save_local("faiss_index_knowledge")

print("✅ SUNO 知识库已构建并保存为 faiss_index_knowledge")
