import os
from typing import Any, Optional

class QianFanBaseAPI:
    """百度千帆API基类"""
    
    def __init__(self):
        self.api_key = os.getenv("QIANFAN_API_KEY")
        if not self.api_key:
            raise ValueError("请设置环境变量 QIANFAN_API_KEY")
        
        self.base_url = "https://qianfan.baidubce.com/v2/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def get_model_list(self) -> Optional[Any]:
        """获取支持的模型列表"""
        return [
            {
                "name": "qianfan/ernie-4.5-8k-preview",
                "description": "ERNIE 4.5最新版本，支持8K上下文，支持function call"
            },
            {
                "name": "qianfan/ernie-4.0-8k-latest",
                "description": "ERNIE 4.0最新版本，支持8K上下文，支持function call"
            },
            {
                "name": "qianfan/ernie-4.0-turbo-8k-latest",
                "description": "ERNIE 4.0 Turbo版本，支持8K上下文，支持function call"
            },
            {
                "name": "qianfan/ernie-4.0-turbo-128k",
                "description": "ERNIE 4.0 Turbo长文本版本，支持128K上下文，支持function call"
            },
            {
                "name": "qianfan/ernie-speed-8k",
                "description": "ERNIE Speed快速版本，支持8K上下文"
            },
            {
                "name": "qianfan/ernie-speed-128k",
                "description": "ERNIE Speed长文本版本，支持128K上下文"
            },
            {
                "name": "qianfan/deepseek-v3",
                "description": "DeepSeek Chat V3版本，支持8K上下文"
            },
            {
                "name": "qianfan/deepseek-r1",
                "description": "DeepSeek R1推理模型，支持8K上下文"
            },
            {
                "name": "qianfan/qwq-32b",
                "description": "QwQ-32B模型，支持8K上下文"
            },
            {
                "name": "qianfan/deepseek-vl2",
                "description": "deepseek vl2 多模态版本，支持8K上下文"
            },
            {
                "name": "qianfan/deepseek-vl2-small",
                "description": "deepseek vl2-small 多模态版本，支持8K上下文"
            },
            {
                "name": "qianfan/qwen2.5-vl-7b-instruct",
                "description": "qwen2.5-7b 多模态模型，支持8K上下文"
            },
            {
                "name": "qianfan/ernie-4.5-8k-preview",
                "description": "ernie-4.5 多模态模型，支持8K上下文"
            }
        ]