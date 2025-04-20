from typing import Dict, List, Optional, Any, Union, Tuple
from util.mylog import logger
from myhuggingface.huggingface_base import HuggingFaceBase
from util.util import split_model_name
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np

class HuggingFaceRerankerAPI(HuggingFaceBase):
    """HuggingFace 本地重排模型 API 调用封装"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        device: str = "cpu",
        use_auth_token: Optional[str] = None,
        max_length: int = 512
    ) -> None:
        self.model_path = None
        if not model_path:
            self.model_path = "jinaai/jina-reranker-v2-base-multilingual"  # 默认使用中文重排模型
            
        super().__init__(self.model_path, cache_dir, device, use_auth_token)
        self.max_length = max_length
        # 延迟加载模型
        self._model = None
        self._tokenizer = None
    
    def _load_model_and_tokenizer(self):
        """加载模型和分词器"""
        if self._model is None or self._tokenizer is None:
            logger.info(f"正在加载重排模型: {self.model_path}")
            
            try:
                # 加载分词器
                tokenizer_kwargs = {
                    "cache_dir": self.cache_dir,
                    "use_auth_token": self.use_auth_token,
                    "trust_remote_code": True
                }
                
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    **tokenizer_kwargs
                )
                
                # 加载模型并指定数据类型
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_path,
                    cache_dir=self.cache_dir,
                    use_auth_token=self.use_auth_token,
                    device_map=self.device,
                    trust_remote_code=True,
                    torch_dtype=torch.float32  # 强制使用 float32
                )
                
                # 确保模型使用 float32
                self._model = self._model.to(torch.float32)
                
                logger.info(f"重排模型加载完成: {self.model_path}")
                
            except Exception as e:
                logger.error(f"加载重排模型或分词器失败: {str(e)}")
                raise e
    
    @property
    def model(self):
        """获取模型"""
        if self._model is None:
            self._load_model_and_tokenizer()
        return self._model
    
    @property
    def tokenizer(self):
        """获取分词器"""
        if self._tokenizer is None:
            self._load_model_and_tokenizer()
        return self._tokenizer
            
    def get_model_list(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """获取可用模型列表"""
        return [
            {
                'name': 'huggingface/jinaai/jina-reranker-v2-base-multilingual',
                'description': 'jina-reranker 重排模型'
            },
        ]
    
    def rerank(
        self, 
        query: str, 
        documents: List[str], 
        top_k: Optional[int] = None,
        with_score: bool = False
    ) -> List[Tuple[str, float]]:
        """对文档进行重排序
        
        Args:
            query: 查询文本
            documents: 要重排的文档列表
            top_k: 返回的文档数量，默认返回所有文档
            
        Returns:
            List[Tuple[str, float]]: 重排后的文档和分数列表，按分数降序排列
        """
        if not documents:
            return []
            
        try:
            model = self.model
            tokenizer = self.tokenizer
            
            # 准备输入
            features = []
            for doc in documents:
                # 对于重排模型，通常需要将查询和文档一起输入
                inputs = tokenizer(
                    query, 
                    doc, 
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                
                # 将输入移动到正确的设备
                if self.device != "cpu":
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                features.append(inputs)
            
            # 计算分数
            scores = []
            for feature in features:
                with torch.no_grad():
                    outputs = model(**feature)
                    
                # 获取相关性分数
                if hasattr(outputs, "logits"):
                    # 对于二分类模型，通常使用第二个类别的分数作为相关性分数
                    if outputs.logits.shape[1] == 2:
                        score = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()[0]
                    else:
                        # 对于回归模型，直接使用输出分数
                        score = outputs.logits.cpu().numpy()[0][0]
                else:
                    # 如果没有 logits 属性，尝试直接使用输出
                    score = outputs[0].cpu().numpy()[0]
                    
                scores.append(score)
            
            # 将文档和分数组合，并按分数降序排序
            doc_score_pairs = list(zip(documents, scores))
            doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # 如果指定了 top_k，则只返回前 top_k 个文档
            if top_k is not None and top_k > 0:
                doc_score_pairs = doc_score_pairs[:top_k]

            if not with_score:
                doc_score_pairs = [pair[0] for pair in doc_score_pairs]
                
            return doc_score_pairs
            
        except Exception as e:
            logger.error(f"重排文档失败: {str(e)}")
            # 出错时返回原始顺序，但没有分数
            return [(doc, 0.0) for doc in documents]

if __name__ == "__main__":
    # 测试 HuggingFace 重排模型
    try:
        # 使用 API 直接调用
        api = HuggingFaceRerankerAPI(device='cpu')
        
        # 测试查询和文档
        query = "深度学习如何处理自然语言"
        documents = [
            "深度学习模型如BERT和GPT在自然语言处理领域取得了巨大突破。",
            "自然语言处理是人工智能的一个分支，专注于计算机与人类语言的交互。",
            "机器学习算法可以用于图像识别和计算机视觉任务。",
            "深度神经网络通过多层结构学习语言的复杂表示，从而理解自然语言。",
            "数据库系统用于存储和管理大量结构化数据。"
        ]
        
        # 重排文档
        reranked_docs = api.rerank(query, documents, with_score=True)
        
        # 打印结果
        logger.info(f"查询: {query}")
        logger.info("重排结果:")
        for i, (doc, score) in enumerate(reranked_docs):
            logger.info(f"{i+1}. 分数: {score:.4f}, 文档: {doc}")
        
    except Exception as e:
        logger.error(f"测试失败: {str(e)}")