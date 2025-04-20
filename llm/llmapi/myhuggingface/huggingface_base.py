import os
from typing import Optional, List, Dict, Any
from util.mylog import logger
from transformers import AutoConfig

class HuggingFaceBase:
    """HuggingFace 本地模型基础配置类"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        device: str = "cpu",
        use_auth_token: Optional[str] = None,
    ) -> None:
        """初始化 HuggingFace 基础配置
        
        Args:
            model_path: 模型本地路径，如果不提供则使用默认模型
            cache_dir: 模型缓存目录，如果不提供则使用默认缓存目录
            device: 推理设备，默认为 "cuda"，可选 "cpu"
            use_auth_token: HuggingFace Hub 认证令牌，用于下载私有模型
        """
        # 设置模型路径
        if not model_path:
            logger.warning("未设置 HuggingFace，将使用默认模型")
            raise ValueError("未设置 HuggingFace 模型路径")

        # 设置缓存目录
        self.cache_dir = cache_dir or os.getenv("HUGGINGFACE_CACHE_DIR")
        # 设置设备
        self.device = device
        # 设置认证令牌
        self.use_auth_token = use_auth_token or os.getenv("HUGGINGFACE_TOKEN")
        logger.info(f"HuggingFace 配置已初始化，模型路径: {self.model_path}")
    
    def validate_model_path(self) -> bool:
        """验证模型路径是否有效"""
        try:
            config = AutoConfig.from_pretrained(self.model_path, cache_dir=self.cache_dir)
            return config is not None
        except Exception as e:
            logger.error(f"模型路径验证失败: {str(e)}")
            return False
            
    @property
    def default_model(self) -> str:
        """获取默认模型名称
        
        Returns:
            str: 默认模型名称
        """
        return "Qwen/Qwen2.5-0.5B-Instruct"
        
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息
        
        Returns:
            Dict[str, Any]: 模型信息
        """
        try:
            config = AutoConfig.from_pretrained(self.model_path, cache_dir=self.cache_dir)
            return {
                "model_type": config.model_type,
                "vocab_size": config.vocab_size,
                "hidden_size": config.hidden_size,
                "num_hidden_layers": config.num_hidden_layers,
                "num_attention_heads": config.num_attention_heads
            }
        except Exception as e:
            logger.error(f"获取模型信息失败: {str(e)}")
            return {"error": str(e)}

if __name__ == "__main__":
    # 测试基础配置
    try:
        base = HuggingFaceBase("Qwen/Qwen2.5-0.5B-Instruct")
        logger.info(f"默认模型: {base.default_model}")
        
        if base.validate_model_path():
            logger.info("模型路径验证成功")
            logger.info(f"模型信息: {base.get_model_info()}")
        else:
            logger.error("模型路径验证失败")
            
    except Exception as e:
        logger.error(f"初始化失败: {str(e)}")