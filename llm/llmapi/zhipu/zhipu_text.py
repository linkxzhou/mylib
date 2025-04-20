from util.mylog import logger
import requests
import json
import os
from typing import Optional, Dict, Any, Generator
import base64
from PIL import Image
import io
from zhipu.zhipu_base import ZhipuBaseAPI

# 
class ZhipuTextAPI(ZhipuBaseAPI):
    def __init__(self):
        super().__init__()
        self.base_url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

    def get_model_list(self) -> Optional[Any]:
        return [
            {
                "name": "zhipu/glm-4-flash",
                "description": "GLM-4 Flash 快速版本（免费模型）"
            },
            {
                "name": "zhipu/glm-4v-flash",
                "description": "GLM-4v Flash 多模态版本（免费模型）"
            },
            {
                "name": "zhipu/glm-4-plus",
                "description": "GLM-4 Plus 模型"
            },
            {
                "name": "zhipu/glm-4",
                "description": "GLM-4 标准版本"
            },
            {
                "name": "zhipu/glm-4-air",
                "description": "GLM-4 Air 轻量版"
            },
            {
                "name": "zhipu/glm-4-long",
                "description": "GLM-4 长文本版本"
            },
            
            {
                "name": "zhipu/glm-4v",
                "description": "GLM-4V 多模态标准版本"
            }
        ]

    def image_to_base64(self, image) -> Optional[str]:
        if image is None:
            return None
        if isinstance(image, str):  # 如果是文件路径
            with open(image, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        # 如果是PIL Image
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def generate_text_with_image(
        self,
        prompt: str,
        image,
        model: str = "glm-4v",
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_tokens: int = None,
        stream: bool = False,
        request_id: str = None,
        do_sample: bool = True,
        user_id: str = None
    ) -> Dict[str, Any]:
        """
        生成带图片的文本响应
        
        Args:
            prompt: 输入提示文本
            image: 输入图片(可以是文件路径或PIL Image对象)
            model: 模型名称，默认"glm-4v"
            temperature: 温度参数，取值区间[0.0,1.0]
            top_p: 采样参数，取值区间[0.0,1.0]
            max_tokens: 最大输出token数
            stream: 是否使用流式响应，默认False
            request_id: 请求ID
            do_sample: 是否启用采样策略，默认True
            user_id: 用户唯一标识
        """
        image_base64 = self.image_to_base64(image)
        if not image_base64:
            logger.error("图片处理失败")
            return {"error": "图片处理失败"}

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_base64
                        }
                    }
                ]
            }
        ]
        
        data = {
            "model": model,
            "messages": messages,
            "stream": stream
        }

        # 添加可选参数
        if temperature is not None:
            data["temperature"] = temperature
        if top_p is not None:
            data["top_p"] = top_p
        if max_tokens is not None:
            data["max_tokens"] = max_tokens
        if request_id is not None:
            data["request_id"] = request_id
        if do_sample is not None:
            data["do_sample"] = do_sample
        if user_id is not None:
            data["user_id"] = user_id

        try:
            logger.info(f"开始生成图文响应，模型: {model}")
            logger.debug(f"请求参数: {json.dumps(data, ensure_ascii=False, indent=2)}")
            
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=data,
                timeout=(5, 600),
                stream=stream
            )
            response.raise_for_status()
            
            if stream:
                logger.info("开始接收流式响应")
                return self._handle_stream_response(response)
            else:
                response_json = response.json()
                logger.info("图文响应生成成功")
                logger.debug(f"响应结果: {json.dumps(response_json, ensure_ascii=False, indent=2)}")
                return response_json
            
        except requests.exceptions.HTTPError as http_err:
            error_msg = self._handle_http_error(response.status_code, http_err)
            logger.error(f"HTTP错误: {error_msg}")
            return {"error": error_msg}
            
        except Exception as err:
            logger.error(f"请求异常: {str(err)}")
            return {"error": str(err)}

    def generate_text(
        self, 
        prompt: str,
        model: str = "glm-4",
        stream: bool = False,
        temperature: float = None,
        top_p: float = None,
        max_tokens: int = None,
        request_id: str = None,
        do_sample: bool = True,
        user_id: str = None,
        stop: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        生成文本响应
        
        Args:
            prompt: 输入提示文本
            model: 模型名称，默认"glm-4"
            stream: 是否使用流式响应，默认False
            temperature: 温度参数，取值区间[0.0,1.0]
            top_p: 采样参数，取值区间[0.0,1.0]
            max_tokens: 最大输出token数
            request_id: 请求ID
            do_sample: 是否启用采样策略，默认True
            user_id: 用户唯一标识
            stop: 自定义结束生成的字符串
            
        Returns:
            Dict: API响应结果或生成器
        """
        data = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": stream
        }

        # 添加可选参数
        if temperature is not None:
            data["temperature"] = temperature
        if top_p is not None:
            data["top_p"] = top_p
        if max_tokens is not None:
            data["max_tokens"] = max_tokens
        if request_id is not None:
            data["request_id"] = request_id
        if do_sample is not None:
            data["do_sample"] = do_sample
        if user_id is not None:
            data["user_id"] = user_id
        if stop is not None:
            data["stop"] = [stop]

        try:
            logger.info(f"开始生成文本响应，模型: {model}, 流式响应: {stream}")
            logger.debug(f"请求参数: {json.dumps(data, ensure_ascii=False, indent=2)}")
            
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=data,
                timeout=(5, 600),
                stream=stream
            )
            response.raise_for_status()

            if stream:
                logger.info("开始接收流式响应")
                return self._handle_stream_response(response)
            else:
                response_json = response.json()
                logger.info("文本响应生成成功")
                logger.debug(f"响应结果: {json.dumps(response_json, ensure_ascii=False, indent=2)}")
                return response_json

        except requests.exceptions.HTTPError as http_err:
            error_msg = self._handle_http_error(response.status_code, http_err)
            logger.error(f"HTTP错误: {error_msg}")
            return {"error": error_msg}
            
        except Exception as err:
            logger.error(f"请求异常: {str(err)}")
            return {"error": str(err)}

    def _handle_stream_response(self, response) -> Generator:
        """处理流式响应"""
        for line in response.iter_lines():
            if line:
                if line.strip() == b"data: [DONE]":
                    logger.debug("流式响应结束")
                    break
                if line.startswith(b"data: "):
                    json_str = line[6:].decode('utf-8')
                    yield json.loads(json_str)

    def _handle_http_error(self, status_code: int, error: Exception) -> str:
        """处理HTTP错误并返回适当的错误消息"""
        error_messages = {
            400: "请求格式有误",
            401: "鉴权不通过",
            429: "请求并发数超过限额",
            500: "内部错误"
        }
        return error_messages.get(status_code, f"HTTP error occurred: {error}")

    def generate_text_with_conversation(
        self,
        messages: list,
        model: str = "glm-4",
        stream: bool = False,
        temperature: float = None,
        top_p: float = None,
        max_tokens: int = None,
        request_id: str = None,
        do_sample: bool = True,
        user_id: str = None,
        stop: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        使用对话历史生成文本响应
        
        Args:
            messages: 对话历史消息列表，格式为[{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]
            model: 模型名称，默认"glm-4"
            stream: 是否使用流式响应，默认False
            temperature: 温度参数，取值区间[0.0,1.0]
            top_p: 采样参数，取值区间[0.0,1.0]
            max_tokens: 最大输出token数
            request_id: 请求ID
            do_sample: 是否启用采样策略，默认True
            user_id: 用户唯一标识
            stop: 自定义结束生成的字符串
            
        Returns:
            Dict: API响应结果或生成器
        """
        data = {
            "model": model,
            "messages": messages,
            "stream": stream
        }

        # 添加可选参数
        if temperature is not None:
            data["temperature"] = temperature
        if top_p is not None:
            data["top_p"] = top_p
        if max_tokens is not None:
            data["max_tokens"] = max_tokens
        if request_id is not None:
            data["request_id"] = request_id
        if do_sample is not None:
            data["do_sample"] = do_sample
        if user_id is not None:
            data["user_id"] = user_id
        if stop is not None:
            data["stop"] = [stop]

        try:
            logger.info(f"开始生成对话响应，模型: {model}, 流式响应: {stream}")
            logger.debug(f"请求参数: {json.dumps(data, ensure_ascii=False, indent=2)}")
            
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=data,
                timeout=(5, 600),
                stream=stream
            )
            response.raise_for_status()

            if stream:
                logger.info("开始接收流式响应")
                return self._handle_stream_response(response)
            else:
                response_json = response.json()
                logger.info("对话响应生成成功")
                logger.debug(f"响应结果: {json.dumps(response_json, ensure_ascii=False, indent=2)}")
                return response_json

        except requests.exceptions.HTTPError as http_err:
            error_msg = self._handle_http_error(response.status_code, http_err)
            logger.error(f"HTTP错误: {error_msg}")
            return {"error": error_msg}
            
        except Exception as err:
            logger.error(f"请求异常: {str(err)}")
            return {"error": str(err)}

if __name__ == "__main__":
    zhipu_api = ZhipuTextAPI()
    
    # 测试普通响应
    prompt = "帮我写一段go版本的快速排序代码"
    response = zhipu_api.generate_text(prompt)
    
    if "error" in response:
        logger.error(f"生成失败: {response['error']}")
    else:
        logger.debug(f"完整响应: {json.dumps(response, ensure_ascii=False, indent=2)}")
        if "choices" in response:
            content = response["choices"][0]["message"]["content"]
            logger.info(f"生成的文本:\n{content}")