from typing import Optional
from util.mylog import logger
from myhuggingface.huggingface_reranker import HuggingFaceRerankerAPI

class RerankerFactory:
    """
    Reranker 工厂类，用于创建不同的 Reranker 实例
    """
    
    @staticmethod
    def create(
        reranker_type: str = "huggingface",
        api_key: Optional[str] = None,
        **kwargs
    ) -> any:
        """
        创建 Reranker 实例
        
        Args:
            reranker_type: Reranker类型，目前支持：huggingface
            api_key: API密钥
            **kwargs: 其他参数
            
        Returns:
            any实例
            
        Raises:
            ValueError: 不支持的Reranker类型
        """
        if reranker_type.lower() == "huggingface":
            return HuggingFaceRerankerAPI(**kwargs)
        else:
            raise ValueError(f"不支持的Reranker类型: {reranker_type}")

if __name__ == "__main__":
    # 创建Embedding实例
    rerankers = [(name, RerankerFactory.create(name)) for name in ["huggingface"]] 
    # 测试重排
    query = "深度学习如何处理自然语言"
    documents = [
        "深度学习模型如BERT和GPT在自然语言处理领域取得了巨大突破。",
        "自然语言处理是人工智能的一个分支，专注于计算机与人类语言的交互。",
        "机器学习算法可以用于图像识别和计算机视觉任务。"
    ]

    # 测试向量化
    for name, reranker in rerankers:
        try:
            logger.info(f"====== 使用 {name} Reranker")
            for doc in documents:
                logger.info(f"文档: {doc}")
                reranked_docs = reranker.rerank(query, documents, with_score=True)
                logger.info("重排结果:")
                for i, (doc, score) in enumerate(reranked_docs):
                    logger.info(f"{i+1}. 分数: {score:.4f}, 文档: {doc}")
        except Exception as e:
            logger.error(f"测试 {name} Reranker 失败: {str(e)}")
    
    