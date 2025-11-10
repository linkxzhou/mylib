from typing import Optional, Dict
from myhuggingface.huggingface_embedding import HuggingFaceEmbedding
from util.mylog import logger
from util.base import BaseEmbedding

class EmbeddingFactory:
    """
    Embedding 工厂类，用于创建不同的 Embedding 实例
    """
    _instances: Dict[str, BaseEmbedding] = {}

    @staticmethod
    def create(
        embedding_type: str = "huggingface",
        api_key: Optional[str] = None,
        **kwargs
    ) -> BaseEmbedding:
        """
        创建 Embedding 实例（按类型单例：相同 embedding_type 仅首次初始化）
        """
        t = (embedding_type or "").lower().strip()
        if t in EmbeddingFactory._instances:
            return EmbeddingFactory._instances[t]

        if t == "huggingface":
            instance = HuggingFaceEmbedding(**kwargs)
            EmbeddingFactory._instances[t] = instance
            return instance
        else:
            raise ValueError(f"不支持的Embedding类型: {embedding_type}")

if __name__ == "__main__":
    # 创建Embedding实例
    embeddings = [(name, EmbeddingFactory.create(name)) for name in ["huggingface"]] 
    # 测试向量化
    for name, embedding in embeddings:
        try:
            logger.info(f"====== 使用 {name} Embedding")
            vector = embedding.embed_query("这是一段测试文本")
            logger.info(f"向量维度: {len(vector)}")
        except Exception as e:
            logger.error(f"测试 {name} Embedding 失败: {str(e)}")
   