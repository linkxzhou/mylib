from typing import Dict, List, Optional, Any, Union
from util.mylog import logger
from llmapi.myhuggingface.huggingface_base import HuggingFaceBase
from util.util import split_model_name
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import threading
from queue import Queue
import time
import os

class HuggingFaceTextAPI(HuggingFaceBase):
    """HuggingFace 本地模型 API 调用封装"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        device: str = "cuda",
        use_auth_token: Optional[str] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False
    ) -> None:
        super().__init__(model_path, cache_dir, device, use_auth_token)
        
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        
        # 延迟加载模型
        self._model = None
        self._tokenizer = None
        self._chat = self.Chat(self)
    
    def _load_model_and_tokenizer(self):
        """加载模型和分词器"""
        if self._model is None or self._tokenizer is None:
            logger.info(f"正在加载模型: {self.model_path}")
            
            try:
                # 加载分词器，使用更安全的方式
                tokenizer_kwargs = {
                    "cache_dir": self.cache_dir,
                    "use_auth_token": self.use_auth_token,
                    "use_fast": False,  # 使用慢速分词器以避免转换问题
                    "trust_remote_code": True  # 允许执行远程代码以支持更多模型
                }
                
                # 检查是否存在本地分词器文件
                if os.path.isdir(self.model_path):
                    # 检查是否有 tokenizer.model 文件（SentencePiece 模型）
                    if os.path.exists(os.path.join(self.model_path, "tokenizer.model")):
                        logger.info("检测到 SentencePiece 分词器模型文件")
                        tokenizer_kwargs["use_fast"] = False
                
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    **tokenizer_kwargs
                )
                
                # 确保分词器有 pad_token
                if self._tokenizer.pad_token is None:
                    logger.info("分词器没有 pad_token，使用 eos_token 作为 pad_token")
                    self._tokenizer.pad_token = self._tokenizer.eos_token
                
                # 设置量化参数
                quantization_config = {}
                if self.load_in_8bit:
                    quantization_config["load_in_8bit"] = True
                elif self.load_in_4bit:
                    quantization_config["load_in_4bit"] = True
                    quantization_config["bnb_4bit_compute_dtype"] = torch.float16
                
                # 加载模型
                self._model = AutoModelForCausallmapi.from_pretrained(
                    self.model_path,
                    cache_dir=self.cache_dir,
                    use_auth_token=self.use_auth_token,
                    device_map=self.device,
                    torch_dtype=torch.float16,
                    trust_remote_code=True,  # 允许执行远程代码以支持更多模型
                    **quantization_config
                )
                
                logger.info(f"模型加载完成: {self.model_path}")
                
            except Exception as e:
                logger.error(f"加载模型或分词器失败: {str(e)}")
                raise e
    
    @property
    def model(self):
        """获取模型"""
        if self._model is None:
            self._load_model_and_tokenizer()
        return self._model
    
    @property
    def tokenizer(self):
        """获取分词器"""
        if self._tokenizer is None:
            self._load_model_and_tokenizer()
        return self._tokenizer
            
    def get_model_list(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """获取可用模型列表"""
        return [
            {
                'name': 'huggingface/Qwen/Qwen2.5-0.5B-Instruct',
                'description': 'Qwen2.5-0.5B-Instruct',
            },
        ]

    @property
    def chat(self):
        """获取聊天接口"""
        return self._chat

    class Chat:
        """聊天相关接口封装"""
        
        def __init__(self, api):
            self.api = api
            
        def create(
            self,
            messages: List[Dict[str, str]],
            model: Optional[str] = None,
            temperature: Optional[float] = 0.7,
            top_p: Optional[float] = 0.9,
            max_tokens: Optional[int] = 2048,
            stop: Optional[Union[str, List[str]]] = None,
            **kwargs: Any
        ) -> Dict[str, Any]:
            try:
                # 如果提供了模型名称，则更新模型路径
                if model:
                    _, model_name = split_model_name(model)
                    self.api.model_path = model_name
                    # 重置模型和分词器，以便重新加载
                    self.api._model = None
                    self.api._tokenizer = None
                
                # 确保模型和分词器已加载
                try:
                    model = self.api.model
                    tokenizer = self.api.tokenizer
                except Exception as e:
                    return {"error": f"模型加载失败: {str(e)}"}
                
                # 构建提示文本
                prompt = self._build_prompt(messages)
                
                # 记录请求参数
                logger.debug(f"HuggingFace 请求参数: prompt={prompt[:100]}..., temperature={temperature}, top_p={top_p}, max_tokens={max_tokens}")
                
                try:
                    # 编码输入
                    inputs = tokenizer(prompt, return_tensors="pt")
                    
                    # 将输入移动到正确的设备
                    if self.api.device != "cpu":
                        inputs = {k: v.to(self.api.device) for k, v in inputs.items()}
                    
                    input_ids = inputs["input_ids"]
                    
                    # 设置生成参数
                    gen_kwargs = {
                        "input_ids": input_ids,
                        "max_new_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        "do_sample": temperature > 0,
                        "pad_token_id": tokenizer.pad_token_id
                    }
                    
                    # 处理停止词
                    if stop:
                        if isinstance(stop, str):
                            stop = [stop]
                        try:
                            stop_token_ids = [tokenizer.encode(s, add_special_tokens=False)[-1] for s in stop]
                            gen_kwargs["eos_token_id"] = stop_token_ids
                        except Exception as e:
                            logger.warning(f"设置停止词失败: {str(e)}")
                    
                    # 非流式生成 - 使用兼容性更好的方式
                    with torch.no_grad():
                        try:
                            # 尝试使用新版本 API
                            output_ids = model.generate(**gen_kwargs)
                        except AttributeError as e:
                            if "'DynamicCache' object has no attribute 'get_max_length'" in str(e):
                                logger.warning("检测到 DynamicCache 兼容性问题，尝试使用替代方法")
                                # 使用不依赖 DynamicCache.get_max_length 的方式
                                gen_kwargs["use_cache"] = True
                                gen_kwargs["past_key_values"] = None
                                output_ids = model.generate(**gen_kwargs)
                            else:
                                raise e
                    
                    # 解码输出
                    output_text = tokenizer.decode(output_ids[0][len(input_ids[0]):], skip_special_tokens=True)
                    
                    # 记录处理后的响应
                    logger.info(f"HuggingFace 响应内容: {output_text[:100]}...")
                    
                    # 构建响应格式
                    result = {
                        "choices": [{
                            "message": {
                                "content": output_text
                            }
                        }]
                    }
                    
                    return result
                
                except Exception as e:
                    error_msg = f"模型推理失败: {str(e)}"
                    logger.error(error_msg)
                    return {"error": error_msg}
                    
            except Exception as e:
                error_msg = f"HuggingFace 模型调用失败: {str(e)}"
                logger.error(error_msg)
                return {"error": error_msg}
        
        def _build_prompt(self, messages: List[Dict[str, str]]) -> str:
            """构建提示文本"""
            prompt = ""
            for message in messages:
                role = message.get("role", "user")
                content = message.get("content", "")
                
                if role == "system":
                    prompt += f"System: {content}\n\n"
                elif role == "user":
                    prompt += f"User: {content}\n\n"
                elif role == "assistant":
                    prompt += f"Assistant: {content}\n\n"
            
            # 添加最后的助手提示
            prompt += "Assistant: "
            return prompt

if __name__ == "__main__":
    # 测试 HuggingFace API
    api = HuggingFaceTextAPI(device='cpu')
    
    # 测试获取模型列表
    models = api.get_model_list()
    logger.info(f"可用模型: {models}")
    
    # 测试聊天功能
    response = api.chat.create(
        messages=[{"role": "user", "content": "你好，请介绍一下你自己"}],
        temperature=0.7,
        max_tokens=100
    )
    logger.info(f"聊天响应: {response}")