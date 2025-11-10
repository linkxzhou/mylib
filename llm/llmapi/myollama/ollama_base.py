import os
from typing import Optional
from util.mylog import logger
import requests

class OllamaBase:
    """Ollama API 基础配置类"""
    
    def __init__(
        self,
        api_base: Optional[str] = None,
    ) -> None:
        """初始化 Ollama 基础配置
        
        Args:
            api_base: Ollama API 基础 URL，如果不提供则使用默认值
        """
        # 设置基础 URL
        self.api_base = api_base or os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
        logger.info(f"Ollama API 配置已初始化，基础 URL: {self.api_base}")
    
    def validate_connection(self) -> bool:
        """验证与 Ollama 服务的连接
        
        Returns:
            bool: 连接是否有效
        """
        try:
            response = requests.get(f"{self.api_base}/api/version")
            if response.status_code == 200:
                version_info = response.json()
                logger.info(f"Ollama 服务连接成功，版本: {version_info.get('version', '未知')}")
                return True
            else:
                logger.error(f"Ollama 服务连接失败，状态码: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Ollama 服务连接异常: {str(e)}")
            return False
            
    @property
    def default_model(self) -> str:
        """获取默认模型名称
        
        Returns:
            str: 默认模型名称
        """
        return "llama3.2:1b"

if __name__ == "__main__":
    # 测试基础配置
    try:
        base = OllamaBase()
        if base.validate_connection():
            logger.info("Ollama 服务连接验证成功")
        else:
            logger.error("Ollama 服务连接验证失败")
            
    except Exception as e:
        logger.error(f"初始化失败: {str(e)}")