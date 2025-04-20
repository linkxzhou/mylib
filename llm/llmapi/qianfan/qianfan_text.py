import os
import json
import requests
from typing import Dict, Any, Optional, List
from util.mylog import logger
from util.util import image_to_base64
from llmapi.qianfan.qianfan_base import QianFanBaseAPI

class QianFanTextAPI(QianFanBaseAPI):
    """百度千帆API客户端实现"""
    
    class Chat:
        def __init__(self, api):
            self.api = api
            
        def create(
            self,
            model: str,
            messages: List[Dict[str, str]],
            temperature: Optional[float] = 0.6,
            top_p: Optional[float] = 0.9,
            max_tokens: Optional[int] = None,
            stop: Optional[str] = None,
            image: Optional[str] = None,
        ) -> Dict[str, Any]:
            """创建聊天补全"""
            if image is not None:
                image_prompt = image_to_base64(image)
                # 修改消息格式以支持图像
                new_messages = []
                for msg in messages:
                    if msg["role"] == "user":
                        new_messages.append({
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": msg["content"]
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": image_prompt
                                    }
                                }
                            ]
                        })
                    else:
                        new_messages.append(msg)
                messages = new_messages
            
            data = {
                "model": model,
                "messages": messages
            }
            
            if temperature is not None:
                data["temperature"] = temperature
            if top_p is not None:
                data["top_p"] = top_p
            if max_tokens is not None:
                data["max_tokens"] = max_tokens
            if stop:
                data["stop"] = stop
            
            logger.info(f"url: {self.api.base_url}")
            logger.debug(f"headers: {self.api.headers}")
            logger.debug(f"API Request: {json.dumps(data, ensure_ascii=False, indent=2)}")
            response = requests.post(
                self.api.base_url,
                headers=self.api.headers,
                json=data
            )
            
            if response.status_code != 200:
                return {"error": response.text}
                
            return response.json()
            
    @property
    def chat(self):
        """获取chat completions接口"""
        return self.Chat(self)

if __name__ == "__main__":
    # 测试代码
    api = QianFanTextAPI()
    response = api.chat.create(
        model="ernie-3.5-8k",
        messages=[{"role": "user", "content": "你好"}]
    )
    logger.info(f"API Response: {json.dumps(response, ensure_ascii=False, indent=2)}")