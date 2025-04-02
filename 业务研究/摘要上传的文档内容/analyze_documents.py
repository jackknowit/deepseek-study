from langchain.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredMarkdownLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
# 更改这里：使用新的 OllamaLLM 导入
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
import os

class DocumentAnalyzer:
    def __init__(self, model_name="deepseek-r1:1.5b"):
        self.model_name = model_name
        # 更改这里：使用 OllamaLLM 替代原来的 Ollama
        self.llm = OllamaLLM(model=model_name)
        self.embeddings = OllamaEmbeddings(model=model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
    def load_document(self, file_path):
        """加载不同类型的文档"""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif ext == '.txt':
            loader = TextLoader(file_path)
        elif ext == '.csv':
            loader = CSVLoader(file_path)
        elif ext == '.md':
            loader = UnstructuredMarkdownLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
            
        return loader.load()
        
    def process_documents(self, documents):
        """处理文档并创建向量数据库"""
        chunks = self.text_splitter.split_documents(documents)
        
        # 创建向量数据库
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory="./knowledge_base"
        )
        return vectordb
        
    def create_qa_chain(self, vectordb):
        """创建问答链"""
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vectordb.as_retriever(
                search_kwargs={"k": 3}
            )
        )
        
    def analyze_document(self, file_path, question):
        """分析文档并回答问题"""
        # 加载文档
        documents = self.load_document(file_path)
        
        # 处理文档
        vectordb = self.process_documents(documents)
        
        # 创建问答链
        qa_chain = self.create_qa_chain(vectordb)
        
        # 获取答案
        return qa_chain.run(question)

# 使用示例
if __name__ == "__main__":
    analyzer = DocumentAnalyzer()
    
    # 分析单个文档
    response = analyzer.analyze_document(
        "/home/jack/book/医疗信息化-产品资料/郑州中医骨伤病医院无纸化病案归档系统采购项目邀标文件.pdf",
        "根据这份文件，在开发病案系统时， 我应该提供那些查询报表?"
    )
    print(response)
