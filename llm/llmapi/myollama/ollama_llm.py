from typing import Any, List, Mapping, Optional, Dict
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from pydantic import Field
from llmapi.myollama.ollama_text import OllamaTextAPI

class OllamaLLM(LLM):
    """Ollama 大模型的 LangChain LLM 实现"""
    
    client: OllamaTextAPI = Field(default_factory=OllamaTextAPI)
    model_name: str = Field(default="llama3.2:1b")
    temperature: Optional[float] = Field(default=None)
    top_p: Optional[float] = Field(default=None)
    max_tokens: Optional[int] = Field(default=None)
    
    class Config:
        arbitrary_types_allowed = True
    
    @property
    def _llm_type(self) -> str:
        """返回 LLM 类型"""
        return "ollama"
    
    @property
    def _supported_params(self) -> List[str]:
        """Return supported parameters."""
        return ["temperature", "top_p", "max_tokens", "model"]
    
    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling Ollama API."""
        return {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 2048,
            "model": "llama3.2:1b"
        }
        
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Execute LLM call"""
        
        # Remove None values from parameters
        params = {
            "model": self.model_name,
            "stop": stop,
        }
        
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.top_p is not None:
            params["top_p"] = self.top_p
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
            
        # Add any additional parameters
        params.update({k: v for k, v in kwargs.items() if k not in params and v is not None})
        
        # Check if using chat or generate
        if "messages" in kwargs:
            messages = kwargs.pop("messages")
            messages.append({"role": "user", "content": prompt})
            params["messages"] = messages
            response = self.client.chat.create(**params)
        else:
            params["prompt"] = prompt
            response = self.client.chat.generate(**params)
        
        if "error" in response:
            raise ValueError(f"API call error: {response['error']}")
            
        if "choices" not in response:
            raise ValueError("Invalid API response format")
        
        result = response["choices"][0]["message"]["content"]
        return result
        
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """获取模型标识参数"""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens
        }

if __name__ == "__main__":
    from util.mylog import logger
    
    # 测试 Ollama 模型
    llm = OllamaLLM()
    
    # 测试基本生成
    prompt = "请用简短的话介绍一下人工智能"
    logger.info(f"\n测试基本生成\n提示: {prompt}")
    response = llm(prompt)
    logger.info(f"响应: {response}")
    
    # 测试带参数的生成
    prompt = "写一首关于春天的诗"
    logger.info(f"\n测试带参数的生成\n提示: {prompt}")
    response = llm(
        prompt,
        temperature=0.8,
        max_tokens=100
    )
    logger.info(f"响应: {response}")
    
    # 测试聊天历史
    logger.info("\n测试聊天历史")
    messages = [
        {"role": "user", "content": "你好，我想了解一下深度学习"},
        {"role": "assistant", "content": "深度学习是机器学习的一个分支，它使用多层神经网络来学习数据的特征。"},
        {"role": "user", "content": "那神经网络是什么？"}
    ]
    response = llm(
        "",  # 空提示，因为最后一条消息已经包含了问题
        messages=messages
    )
    logger.info(f"响应: {response}")