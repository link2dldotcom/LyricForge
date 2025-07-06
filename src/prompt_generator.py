from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from .rag import SongRAG
import os
from dotenv import load_dotenv

load_dotenv()

class SUNOPromptGenerator:
    def __init__(self):
        self.rag = SongRAG()
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
        self.prompt_template = ChatPromptTemplate.from_template(
            """
            你是一位专业的音乐制作人，擅长为SUNO AI音乐生成器创作高质量提示词。
            根据以下要求和参考资料，创作一个详细的SUNO提示词：

            要求：
            {requirements}

            参考资料：
            {reference_material}

            SUNO提示词格式应该包含：
            1. 音乐风格
            2. 情感/氛围
            3. 乐器/声音特点
            4. 歌词主题
            5. 参考艺术家/歌曲
            6. 速度和节奏特点

            请创作一个完整、详细且富有创意的SUNO提示词，确保能生成高质量的音乐作品。
            """
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

    def generate_prompt(self, user_requirements):
        # 从RAG获取相关参考资料
        reference_material = self.rag.query_rag(user_requirements)

        # 生成提示词
        result = self.chain.run(
            requirements=user_requirements,
            reference_material=reference_material
        )
        return result

    def train_with_feedback(self, user_requirements, generated_prompt, feedback, rating):
        # 将成功案例添加到知识库进行自我训练
        feedback_note = f"用户需求: {user_requirements}\n生成提示词: {generated_prompt}\n反馈: {feedback}\n评分: {rating}\n"
        self.rag.add_document(
            feedback_note,
            metadata={"type": "feedback", "rating": rating}
        )
        return "训练数据已成功添加"