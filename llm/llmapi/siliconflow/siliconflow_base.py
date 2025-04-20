import os
from typing import Optional
from util.mylog import logger

class SiliconFlowBase:
    """SiliconFlow API 基础配置类"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None
    ) -> None:
        """初始化SiliconFlow基础配置
        
        Args:
            api_key: SiliconFlow API 密钥，如果不提供则从环境变量获取
            api_base: SiliconFlow API 基础 URL，如果不提供则使用默认值
        """
        # 设置 API 密钥
        self.api_key = api_key or os.getenv("SILICONFLOW_API_KEY")
        if not self.api_key:
            raise ValueError("未设置SiliconFlow API 密钥")
            
        # 设置基础 URL（可选）
        self.api_base = api_base or os.getenv("SILICONFLOW_API_BASE", "https://api.siliconflow.com/v1")
        logger.info("SiliconFlow API 配置已初始化")
            
    @property
    def default_model(self) -> str:
        """获取默认模型名称
        
        Returns:
            str: 默认模型名称
        """
        return "deepseek-ai/DeepSeek-V3"
        
    @property
    def api_version(self) -> str:
        """获取 API 版本
        
        Returns:
            str: API 版本
        """
        return "v1"

if __name__ == "__main__":
    # 测试基础配置
    try:
        base = SiliconFlowBase()
        logger.info(f"API 版本: {base.api_version}")
        logger.info(f"默认模型: {base.default_model}")
            
    except Exception as e:
        logger.error(f"初始化失败: {str(e)}")