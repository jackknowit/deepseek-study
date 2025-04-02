from langchain.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.embeddings import OllamaEmbeddings
import json
import os

class MedicalRecordQuery:
    def __init__(self, knowledge_base_dir: str = "./medical_knowledge_base"):
        self.knowledge_base_dir = knowledge_base_dir
        self.llm = OllamaLLM(model="deepseek-r1:1.5b")
        self.embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")
        
        # 加载向量数据库
        self.vectordb = Chroma(
            persist_directory=self.knowledge_base_dir,
            embedding_function=self.embeddings
        )
        
        # 创建检索链
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectordb.as_retriever(
                search_kwargs={
                    "k": 5,  # 返回最相关的5个文档片段
                    "filter": None  # 初始无过滤器
                }
            )
        )
        
    def query_by_visit_id(self, visit_id: str, query: str):
        """根据就诊ID查询病历信息"""
        # 设置元数据过滤器
        self.qa_chain.retriever.search_kwargs["filter"] = {
            "visit_id": visit_id
        }
        
        # 构造系统提示词
        system_prompt = f"""
        你是一位专业的医疗顾问。请基于病历内容，对就诊ID为{visit_id}的患者情况进行分析。
        在回答时请：
        1. 先总结患者的基本情况和主要症状
        2. 分析现有检查结果和诊断
        3. 给出具体的治疗建议
        4. 如有需要，提出进一步检查建议
        请确保建议符合医疗规范和伦理要求。
        """
        
        # 组合查询
        full_query = f"{system_prompt}\n\n用户查询：{query}"
        
        return self.qa_chain.run(full_query)

if __name__ == "__main__":
    query_system = MedicalRecordQuery()
    
    while True:
        user_input = input("\n请输入查询（输入'退出'结束）: ")
        if user_input.lower() in ['退出', 'quit', 'exit']:
            break
            
        try:
            response = query_system.query_by_visit_id(
                "a123",  # 指定的就诊ID
                user_input
            )
            print("\n回答：", response)
        except Exception as e:
            print(f"查询出错：{str(e)}")