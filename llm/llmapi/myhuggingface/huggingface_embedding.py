from typing import Dict, List, Optional, Any, Union
from util.mylog import logger
from langchain_community.embeddings import HuggingFaceEmbeddings
from llmapi.myhuggingface.huggingface_base import HuggingFaceBase
from util.base import BaseEmbedding
from util.util import split_model_name
import torch
import numpy as np

class HuggingFaceEmbedding(BaseEmbedding, HuggingFaceBase):
    """HuggingFace 本地嵌入模型 API 调用封装"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        device: str = "cpu",
        use_auth_token: Optional[str] = None,
        max_length: int = None,
        encode_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not model_path:
            model_path = "jinaai/jina-embeddings-v3"  # 默认使用轻量级嵌入模型

        self.model_path = model_path
        BaseEmbedding.__init__(self)
        HuggingFaceBase.__init__(self, model_path, cache_dir, device, use_auth_token)

        self.max_length = max_length
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_path,
            model_kwargs={
                "device": device, 
                "trust_remote_code": True,
            },  # Use "cuda" for GPU
            encode_kwargs={
                "normalize_embeddings": True,
            } # set True to compute cosine similarity
        )
    
    def get_model_list(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """获取可用模型列表"""
        return [
            {
                'name': 'huggingface/jinaai/jina-embeddings-v2-base-code',
                'description': 'jina-embeddings-v2-base-code 嵌入模型 (768维)',
                'dimension': 768
            },
            {
                'name': 'huggingface/jinaai/jina-embeddings-v3',
                'description': 'jinaai/jina-embeddings-v3 嵌入模型 (1024维)',
                'dimension': 1024
            },
            {
                'name': 'huggingface/BAAI/bge-m3',
                'description': 'BAAI/bge-m3 多语言嵌入模型 (1024维)',
                'dimension': 1024
            }
        ]
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """将多个文本转换为嵌入向量
        
        Args:
            texts: 要嵌入的文本列表
            
        Returns:
            List[List[float]]: 嵌入向量列表
        """
        return self.embeddings.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """将单个查询文本转换为嵌入向量
        
        Args:
            text: 要嵌入的文本
            
        Returns:
            List[float]: 嵌入向量
        """
        return self.embeddings.embed_query(text)

if __name__ == "__main__":
    # 使用 API 直接调用
    embeddings_model = HuggingFaceEmbedding()
    
    # 测试单个文本嵌入
    text = "这是一个测试文本，用于生成嵌入向量。"
    embedding = embeddings_model.embed_query(text)
    logger.info(f"单个文本嵌入维度: {len(embedding)}")
    
    # 测试多个文本嵌入
    texts = [
        "这是第一个测试文本。",
        "这是第二个测试文本，内容不同。",
        "这是第三个完全不相关的文本。"
    ]
    embeddings = embeddings_model.embed_documents(texts)
    logger.info(f"多个文本嵌入数量: {len(embeddings)}")
    logger.info(f"向量维度：{embeddings_model.dimension()}")