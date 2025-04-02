from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
import os
import logging
from typing import Dict, List
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalRecordImporter:
    def __init__(self, knowledge_base_dir: str = "./medical_knowledge_base"):
        self.knowledge_base_dir = knowledge_base_dir
        self.embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", "。", "，", " ", ""]
        )
        
    def process_medical_record(self, file_path: str, visit_id: str) -> List[Dict]:
        """处理单个病历文件，添加就诊ID标记"""
        try:
            # 加载文本文件
            loader = TextLoader(file_path)
            documents = loader.load()
            
            # 分割文档
            chunks = self.text_splitter.split_documents(documents)
            
            # 为每个chunk添加就诊ID元数据
            for chunk in chunks:
                chunk.metadata['visit_id'] = visit_id
                
            return chunks
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return []

    def import_records(self, records_dir: str, id_file_mapping: Dict[str, str]):
        """导入多个病历文件"""
        all_documents = []
        
        # 确保知识库目录存在
        os.makedirs(self.knowledge_base_dir, exist_ok=True)
        
        # 处理每个病历文件
        for visit_id, filename in id_file_mapping.items():
            file_path = os.path.join(records_dir, filename)
            if os.path.exists(file_path):
                logger.info(f"Processing record for visit_id {visit_id}")
                documents = self.process_medical_record(file_path, visit_id)
                all_documents.extend(documents)
            else:
                logger.error(f"File not found: {file_path}")
        
        if all_documents:
            # 创建向量数据库
            vectordb = Chroma.from_documents(
                documents=all_documents,
                embedding=self.embeddings,
                persist_directory=self.knowledge_base_dir
            )
            vectordb.persist()
            logger.info(f"Successfully imported {len(all_documents)} document chunks")
            
            # 保存ID映射关系
            with open(os.path.join(self.knowledge_base_dir, 'id_mapping.json'), 'w') as f:
                json.dump(id_file_mapping, f, ensure_ascii=False, indent=2)
        else:
            logger.error("No documents were processed successfully")

if __name__ == "__main__":
    # 示例使用
    records_dir = "./medical_records"
    id_file_mapping = {
        "a123": "patient_a123.txt",
        # 添加更多映射...
    }
    
    importer = MedicalRecordImporter()
    importer.import_records(records_dir, id_file_mapping)