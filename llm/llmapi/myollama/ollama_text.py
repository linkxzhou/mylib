from typing import Dict, List, Optional, Any, Union
from util.mylog import logger
from myollama.ollama_base import OllamaBase
import requests
import json

class OllamaTextAPI(OllamaBase):
    """Ollama API 调用封装"""
    
    def __init__(
        self,
        api_base: Optional[str] = None
    ) -> None:
        super().__init__(api_base)
        self._chat = self.Chat(self)
    
    def get_model_list(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """获取可用模型列表
        
        Args:
            force_refresh: 是否强制刷新缓存
            
        Returns:
            List[Dict[str, Any]]: 模型列表
        """
        try:
            response = requests.get(f"{self.api_base}/api/tags")
            if response.status_code == 200:
                models_data = response.json().get("models", [])
                result = []
                
                for model in models_data:
                    model_name = model.get("name")
                    if model_name:
                        result.append({
                            'name': f'ollama/{model_name}',
                            'description': f'Ollama 本地模型: {model_name}'
                        })
                
                return result
            else:
                logger.error(f"获取模型列表失败，状态码: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"获取模型列表异常: {str(e)}")
            return []

    @property
    def chat(self):
        """获取聊天接口"""
        return self._chat
    
    class Chat:
        """聊天相关接口封装"""
        
        def __init__(self, client):
            self.client = client
        
        def create(
            self,
            messages: List[Dict[str, str]],
            model: str = "llama2",
            temperature: Optional[float] = None,
            top_p: Optional[float] = None,
            max_tokens: Optional[int] = None,
            stop: Optional[Union[str, List[str]]] = None,
            **kwargs: Any
        ) -> Dict[str, Any]:
            """创建聊天完成
            
            Args:
                messages: 消息列表
                model: 模型名称
                temperature: 温度
                top_p: Top-p 采样
                max_tokens: 最大生成令牌数
                stop: 停止序列
                
            Returns:
                Dict[str, Any]: 响应结果
            """
            try:
                # 如果模型名称以 "ollama/" 开头，则去掉前缀
                if model.startswith("ollama/"):
                    model = model[len("ollama/"):]
                
                # 构建请求参数
                params = {
                    "model": model,
                    "messages": messages,
                    "stream": False
                }
                
                if temperature is not None:
                    params["temperature"] = temperature
                if top_p is not None:
                    params["top_p"] = top_p
                if max_tokens is not None:
                    params["max_tokens"] = max_tokens
                if stop is not None:
                    if isinstance(stop, str):
                        params["stop"] = [stop]
                    else:
                        params["stop"] = stop
                
                # 添加其他参数
                for key, value in kwargs.items():
                    params[key] = value
                
                # 记录请求参数
                logger.debug(f"Ollama API 请求参数: {params}")
                
                # 发送请求
                response = requests.post(
                    f"{self.client.api_base}/api/chat",
                    json=params
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # 记录原始响应
                    logger.debug(f"Ollama API 原始响应: {result}")
                    
                    # 转换为与 OpenAI 兼容的格式
                    content = result.get("message", {}).get("content", "")
                    
                    compatible_result = {
                        "choices": [{
                            "message": {
                                "content": content
                            }
                        }]
                    }
                    
                    # 记录处理后的响应
                    logger.info(f"Ollama API 响应内容: {content[:100]}...")
                    
                    return compatible_result
                else:
                    error_msg = f"Ollama API 调用失败，状态码: {response.status_code}, 响应: {response.text}"
                    logger.error(error_msg)
                    return {"error": error_msg}
                
            except Exception as e:
                error_msg = f"Ollama API 调用异常: {str(e)}"
                logger.error(error_msg)
                return {"error": error_msg}
        
        def generate(
            self,
            prompt: str,
            model: str = "llama2",
            temperature: Optional[float] = None,
            top_p: Optional[float] = None,
            max_tokens: Optional[int] = None,
            stop: Optional[Union[str, List[str]]] = None,
            **kwargs: Any
        ) -> Dict[str, Any]:
            """生成文本完成
            
            Args:
                prompt: 提示文本
                model: 模型名称
                temperature: 温度
                top_p: Top-p 采样
                max_tokens: 最大生成令牌数
                stop: 停止序列
                
            Returns:
                Dict[str, Any]: 响应结果
            """
            try:
                # 如果模型名称以 "ollama/" 开头，则去掉前缀
                if model.startswith("ollama/"):
                    model = model[len("ollama/"):]
                
                # 构建请求参数
                params = {
                    "model": model,
                    "prompt": prompt,
                    "stream": False
                }
                
                if temperature is not None:
                    params["temperature"] = temperature
                if top_p is not None:
                    params["top_p"] = top_p
                if max_tokens is not None:
                    params["max_tokens"] = max_tokens
                if stop is not None:
                    if isinstance(stop, str):
                        params["stop"] = [stop]
                    else:
                        params["stop"] = stop
                
                # 添加其他参数
                for key, value in kwargs.items():
                    params[key] = value
                
                # 记录请求参数
                logger.debug(f"Ollama API 请求参数: {params}")
                
                # 发送请求
                response = requests.post(
                    f"{self.client.api_base}/api/generate",
                    json=params
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # 记录原始响应
                    logger.debug(f"Ollama API 原始响应: {result}")
                    
                    # 转换为与 OpenAI 兼容的格式
                    content = result.get("response", "")
                    
                    compatible_result = {
                        "choices": [{
                            "message": {
                                "content": content
                            }
                        }]
                    }
                    
                    # 记录处理后的响应
                    logger.info(f"Ollama API 响应内容: {content[:100]}...")
                    
                    return compatible_result
                else:
                    error_msg = f"Ollama API 调用失败，状态码: {response.status_code}, 响应: {response.text}"
                    logger.error(error_msg)
                    return {"error": error_msg}
                
            except Exception as e:
                error_msg = f"Ollama API 调用异常: {str(e)}"
                logger.error(error_msg)
                return {"error": error_msg}

if __name__ == "__main__":
    # 测试 Ollama API
    api = OllamaTextAPI()
    
    # 测试获取模型列表
    models = api.get_model_list()
    logger.info(f"可用模型: {models}")
    
    # 测试聊天功能
    response = api.chat.create(
        messages=[{"role": "user", "content": "你好，请介绍一下你自己"}],
        model="llama3.2:1b"
    )
    logger.info(f"聊天响应: {response}")
    
    # 测试生成功能
    response = api.chat.generate(
        prompt="写一首关于人工智能的诗",
        model="llama3.2:1b"
    )
    logger.info(f"生成响应: {response}")