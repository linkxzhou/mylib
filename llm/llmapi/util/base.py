## Sentence Window splitting strategy, ref:
#  https://github.com/milvus-io/bootcamp/blob/master/bootcamp/RAG/advanced_rag/sentence_window_with_langchain.ipynb

from typing import List, Optional
from tqdm import tqdm
import numpy as np
from util.mylog import logger

class Chunk:
    def __init__(
        self,
        text: str,
        embedding: Optional[np.ndarray] = None,
        reference: str = "",
        metadata: Optional[dict] = None,
        chunk_id: Optional[str] = None,
    ):
        self.text = text
        self.embedding = embedding if embedding is not None else None
        self.reference = reference
        self.metadata = metadata or {}
        self.chunk_id = chunk_id

class BaseEmbedding:
    def __init__(self):
        self._dimension = None  # 使用下划线表示私有属性
    
    def embed_query(self, text: str) -> List[float]:
        pass

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_chunks(self, chunks: List[Chunk], batch_size=2) -> List[Chunk]:
        texts = [chunk.text for chunk in chunks]
        batch_texts = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
        
        embeddings = []
        for batch_text in tqdm(batch_texts, desc="Embedding chunks"):
            batch_embeddings = self.embed_documents(batch_text)
            embeddings.extend(batch_embeddings)
            logger.info(f"Embedding chunks: {len(batch_embeddings)}, batch_text: {len(batch_text)}")

        logger.info(f"Embedding chunks: {len(embeddings)}, texts: {len(texts)}, batch_texts: {len(batch_texts)}")
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        return chunks

    def dimension(self) -> int:
        """获取嵌入向量的维度。
        
        Returns:
            int: 嵌入向量的维度
        """
        if self._dimension is None:
            sample_embedding = self.embed_query("dimension_test")
            self._dimension = len(sample_embedding)
        
        return self._dimension