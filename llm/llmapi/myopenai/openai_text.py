from typing import Dict, List, Optional, Any
from util.mylog import logger
from myopenai.openai_base import OpenAIBase
from openai import OpenAI

class OpenAITextAPI(OpenAIBase):
    """OpenAI API 调用封装"""
    
    def __init__(self) -> None:
        super().__init__()
        self.client = OpenAI(
            api_key=self.api_key, 
            base_url=self.api_base,
        )
        self._chat = self.Chat(self.client)
            
    def get_model_list(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        # 默认平台：https://openkey.cloud/，可以使用其他的平台替换即可
        return [
            {
                'name': 'openai/gpt-3.5-turbo',
                'description': 'OpenAI 语言模型'
            },
            {
                'name': 'openai/gpt-3.5-turbo-16k',
                'description': 'OpenAI 语言模型'
            },
            {
                'name': 'openai/gpt-4',
                'description': 'OpenAI 语言模型'
            }, 
            {
                'name': 'openai/gpt-4-32k',
                'description': 'OpenAI 语言模型'
            }, 
            {
                'name': 'openai/gpt-4-turbo',
                'description': 'OpenAI 语言模型'
            }, 
            {
                'name': 'openai/gpt-4-vision-preview',
                'description': 'OpenAI 多模态模型'
            }, 
            {
                'name': 'openai/gpt-4o',
                'description': 'OpenAI 语言模型'
            }, 
            {
                'name': 'openai/gpt-4o-mini',
                'description': 'OpenAI 语言模型'
            }, 
            {
                'name': 'openai/gpt-5',
                'description': 'gpt-5 多模态模型'
            },
            {
                'name': 'openai/360GPT_S2_V9',
                'description': 'OpenAI 语言模型'
            },
            {
                'name': 'openai/claude-3-haiku-20240307',
                'description': 'claude3 语言模型（API 便宜）'
            },
            {
                'name': 'openai/claude-3-5-sonnet-20241022',
                'description': 'claude3.5 语言模型'
            },
            {
                'name': 'openai/claude-3-7-sonnet-20250219',
                'description': 'claude3.7 语言模型'
            },
            {
                'name': 'openai/claude-4',
                'description': 'claude4 语言模型'
            },
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
            model: str = "gpt-3.5-turbo",
            temperature: Optional[float] = 0.6,
            top_p: Optional[float] = 0.9,
            max_tokens: Optional[int] = None,
            stop: Optional[str] = None,
            **kwargs: Any
        ) -> Dict[str, Any]:
            try:
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
                logger.debug(f"OpenAI API 请求参数: {params}")
                
                # 使用新版本的 API 调用方式
                response = self.client.chat.completions.create(**params)
                
                # 记录原始响应
                logger.info(f"OpenAI API 原始响应: {response}")
                
                # 转换响应格式
                result = {
                    "choices": [{
                        "message": {
                            "content": response.choices[0].message.content
                        }
                    }]
                }
                
                # 记录处理后的响应
                logger.info(f"OpenAI API 响应内容: {result['choices'][0]['message']['content'][:100]}...")
                
                return result
                
            except Exception as e:
                error_msg = f"OpenAI API 调用失败: {str(e)}"
                logger.error(error_msg)
                return {"error": error_msg}

if __name__ == "__main__":
    # 测试 OpenAI API
    api = OpenAITextAPI()
    
    # 测试获取模型列表
    models = api.get_model_list()
    logger.info(f"可用模型: {models}")
    
    # 测试聊天功能
    response = api.chat.create(
        messages=[{"role": "user", "content": "你好，请介绍一下你自己"}],
        model="gpt-3.5-turbo"
    )
    logger.info(f"聊天响应: {response}")