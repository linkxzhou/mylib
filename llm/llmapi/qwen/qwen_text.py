from typing import Dict, List, Optional, Any
from util.mylog import logger
from llmapi.qwen.qwen_base import QwenBase
from openai import OpenAI
import time

class QwenTextAPI(QwenBase):
    """通义千问 API 调用封装"""
    
    def __init__(self) -> None:
        super().__init__()
        self.client = OpenAI(
            api_key=self.api_key, 
            base_url=self.api_base,
        )
        self._chat = self.Chat(self.client)
            
    def get_model_list(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        return [
            {
                'name': 'qwen/qwen-plus',
                'description': '通义千问 语言模型'
            },
            {
                'name': 'qwen/qwen-coder-plus',
                'description': '通义千问 提供代码支持的语言模型'
            },
            {
                'name': 'qwen/qwen-max-2025-01-25',
                'description': '通义千问系列效果最好的模型，适合复杂、多步骤的任务'
            }
        ]

    @property
    def chat(self):
        """获取chat completions接口"""
        return self._chat

    class Chat:
        """聊天相关接口封装"""
        
        def __init__(self, client):
            self.client = client
            
        def create(
            self,
            messages: List[Dict[str, str]],
            model: str = "qwen-turbo",
            temperature: Optional[float] = 0.6,
            top_p: Optional[float] = 0.9,
            max_tokens: Optional[int] = None,
            stop: Optional[str] = None,
            **kwargs: Any
        ) -> Dict[str, Any]:
            try:
                # 处理模型名称，移除可能的前缀
                if '/' in model:
                    model = model.split('/')[-1]
                
                params = {
                    "model": model,
                    "messages": messages,
                    **kwargs
                }
                
                if temperature is not None:
                    params["temperature"] = temperature
                if top_p is not None:
                    params["top_p"] = top_p
                if max_tokens is not None:
                    params["max_tokens"] = max_tokens
                if stop is not None:
                    params["stop"] = stop
                    
                # 记录请求参数
                logger.debug(f"通义千问 API 请求参数: {params}")
                
                # 调用通义千问 API
                response = self.client.chat.completions.create(**params)
                
                # 记录原始响应
                logger.info(f"通义千问 API 原始响应: {response}")
                
                # 转换为与 OpenAI 兼容的格式
                result = {
                    "choices": [{
                        "message": {
                            "content": response.choices[0].message.content
                        }
                    }]
                }
                
                # 记录处理后的响应
                logger.info(f"通义千问 API 响应内容: {result['choices'][0]['message']['content'][:100]}...")
                
                return result
                
            except Exception as e:
                error_msg = f"通义千问 API 调用失败: {str(e)}"
                logger.error(error_msg)
                return {"error": error_msg}

if __name__ == "__main__":
    # 测试通义千问 API
    api = QwenTextAPI()
    
    # 测试获取模型列表
    models = api.get_model_list()
    logger.info(f"可用模型: {models}")
    
    # 测试聊天功能
    response = api.chat.create(
        messages=[{"role": "user", "content": "你好，请介绍一下你自己"}],
        model="qwen-plus"
    )
    logger.info(f"聊天响应: {response}")