from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

load_dotenv()

class SongRAG:
    def __init__(self, persist_directory="./chroma_db"):
        self.embeddings = OpenAIEmbeddings()
        self.persist_directory = persist_directory
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        self.db = self._load_or_create_db()

    def _load_or_create_db(self):
        if os.path.exists(self.persist_directory):
            return Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        else:
            return self._create_db()

    def _create_db(self):
        # 加载歌词数据
        documents = []
        for filename in os.listdir("./data/song_lyrics"):
            if filename.endswith(".txt"):
                loader = TextLoader(f"./data/song_lyrics/{filename}", encoding="utf-8")
                documents.extend(loader.load())

        # 分割文档
        splits = self.text_splitter.split_documents(documents)

        # 创建向量库
        db = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        db.persist()
        return db

    def query_rag(self, query, k=3):
        docs = self.db.similarity_search(query, k=k)
        return "\n\n".join([doc.page_content for doc in docs])

    def add_document(self, text, metadata=None):
        if metadata is None:
            metadata = {}
        splits = self.text_splitter.split_text(text)
        self.db.add_texts(splits, metadatas=[metadata]*len(splits))
        self.db.persist()