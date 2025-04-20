from typing import Dict, List, Optional, Any, Generator
from util.mylog import logger
from llmapi.siliconflow.siliconflow_base import SiliconFlowBase
import requests
import json
import time

class SiliconFlowTextAPI(SiliconFlowBase):
    """SiliconFlow API 调用封装"""
    
    def __init__(self) -> None:
        super().__init__()
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self._chat = self.Chat(self)
            
    def get_model_list(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """获取可用模型列表"""
        return [
            {
                'name': 'siliconflow/siliconflow-7b',
                'description': 'SiliconFlow 7B 基础模型'
            },
            {
                'name': 'siliconflow/siliconflow-13b',
                'description': 'SiliconFlow 13B 增强模型'
            },
            {
                'name': 'siliconflow/siliconflow-code',
                'description': 'SiliconFlow 代码专用模型'
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
            model: str = "siliconflow-7b",
            temperature: Optional[float] = 0.7,
            top_p: Optional[float] = 0.9,
            max_tokens: Optional[int] = None,
            stop: Optional[str] = None,
            stream: bool = False,
            **kwargs: Any
        ) -> Dict[str, Any]:
            try:
                # 处理模型名称，移除可能的前缀
                if '/' in model:
                    model = model.split('/')[-1]
                
                # 构建请求参数
                params = {
                    "model": model,
                    "messages": messages,
                    "stream": stream,
                    **kwargs
                }
                
                if temperature is not None:
                    params["temperature"] = temperature
                if top_p is not None:
                    params["top_p"] = top_p
                if max_tokens is not None:
                    params["max_tokens"] = max_tokens
                if stop is not None:
                    params["stop"] = [stop] if isinstance(stop, str) else stop
                    
                # 记录请求参数
                logger.debug(f"SiliconFlow API 请求参数: {json.dumps(params, ensure_ascii=False)}")
                
                # 调用SiliconFlow API
                response = requests.post(
                    f"{self.client.api_base}/chat/completions",
                    headers=self.client.headers,
                    json=params,
                    stream=stream,
                    timeout=(5, 300)  # 连接超时5秒，读取超时300秒
                )
                
                response.raise_for_status()
                
                if stream:
                    # 处理流式响应
                    return self._handle_streaming_response(response)
                else:
                    # 处理普通响应
                    response_json = response.json()
                    
                    # 记录原始响应
                    logger.debug(f"SiliconFlow API 原始响应: {json.dumps(response_json, ensure_ascii=False)}")
                    
                    # 转换为与 OpenAI 兼容的格式
                    result = {
                        "id": response_json.get("id", f"chatcmpl-{int(time.time())}"),
                        "object": "chat.completion",
                        "created": response_json.get("created", int(time.time())),
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
                            },
                            "finish_reason": response_json.get("choices", [{}])[0].get("finish_reason", "stop")
                        }],
                        "usage": response_json.get("usage", {
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0
                        })
                    }
                    
                    # 记录处理后的响应
                    content = result["choices"][0]["message"]["content"]
                    logger.info(f"SiliconFlow API 响应内容: {content[:100]}...")
                    
                    return result
                
            except requests.exceptions.RequestException as e:
                error_msg = f"SiliconFlow API 请求失败: {str(e)}"
                logger.error(error_msg)
                return {"error": error_msg}
                
            except Exception as e:
                error_msg = f"SiliconFlow API 调用失败: {str(e)}"
                logger.error(error_msg)
                return {"error": error_msg}
                
        def _handle_streaming_response(self, response) -> Dict[str, Any]:
            """处理流式响应"""
            try:
                # 收集完整的响应
                collected_chunks = []
                collected_messages = []
                
                # 处理SSE流
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            if line == 'data: [DONE]':
                                break
                            
                            chunk = json.loads(line[6:])
                            collected_chunks.append(chunk)
                            
                            if len(chunk['choices']) > 0:
                                chunk_message = chunk['choices'][0].get('delta', {}).get('content', '')
                                if chunk_message:
                                    collected_messages.append(chunk_message)
                
                # 构建完整响应
                full_message = ''.join(collected_messages)
                
                # 构建与OpenAI兼容的响应格式
                result = {
                    "id": collected_chunks[0].get("id", f"chatcmpl-{int(time.time())}") if collected_chunks else f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": collected_chunks[0].get("model", "siliconflow-7b") if collected_chunks else "siliconflow-7b",
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": full_message
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                    }
                }
                
                logger.info(f"SiliconFlow 流式响应完成，总长度: {len(full_message)}")
                return result
                
            except Exception as e:
                error_msg = f"处理流式响应失败: {str(e)}"
                logger.error(error_msg)
                return {"error": error_msg}

    def generate_text(
        self, 
        prompt: str,
        model: str = "siliconflow-7b",
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """生成文本响应
        
        Args:
            prompt: 输入提示文本
            model: 模型名称
            temperature: 温度参数
            top_p: 采样参数
            max_tokens: 最大输出token数
            stream: 是否使用流式响应
            
        Returns:
            Dict: API响应结果
        """
        messages = [{"role": "user", "content": prompt}]
        return self.chat.create(
            messages=messages,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=stream,
            **kwargs
        )
    
    def generate_text_with_conversation(
        self,
        messages: List[Dict[str, str]],
        model: str = "siliconflow-7b",
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """使用对话历史生成文本响应
        
        Args:
            messages: 对话历史消息列表
            model: 模型名称
            temperature: 温度参数
            top_p: 采样参数
            max_tokens: 最大输出token数
            stream: 是否使用流式响应
            
        Returns:
            Dict: API响应结果
        """
        return self.chat.create(
            messages=messages,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=stream,
            **kwargs
        )

if __name__ == "__main__":
    # 测试SiliconFlow API
    api = SiliconFlowTextAPI()
    
    # 测试获取模型列表
    models = api.get_model_list()
    logger.info(f"可用模型: {models}")
    
    # 测试聊天功能
    response = api.chat.create(
        messages=[{"role": "user", "content": "你好，请介绍一下你自己"}],
        model="siliconflow-7b"
    )
    logger.info(f"聊天响应: {response}")