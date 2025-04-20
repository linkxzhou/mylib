from util.mylog import logger
from typing import Any, List, Mapping, Optional, Dict
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from pydantic import Field
from qianfan.qianfan_text import QianFanTextAPI

class QianFanLLM(LLM):
    """百度千帆大模型的 LangChain LLM 实现"""
    
    client: QianFanTextAPI = Field(default_factory=QianFanTextAPI)
    model_name: str = Field(default="deepseek-v3")
    temperature: Optional[float] = Field(default=0.6)
    top_p: Optional[float] = Field(default=0.9)
    max_tokens: Optional[int] = Field(default=None)
    
    class Config:
        arbitrary_types_allowed = True
    
    @property
    def _llm_type(self) -> str:
        """返回 LLM 类型"""
        return "qianfan"

    @property
    def _supported_params(self) -> List[str]:
        """Return supported parameters."""
        return ["temperature", "top_p", "max_tokens", "model"]

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling QianFan API."""
        return {
            "temperature": 0.6,
            "top_p": 0.9,
            "model": "deepseek-v3"
        }
        
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """执行 LLM 调用，支持对话历史和图片输入"""
        messages = kwargs.pop("messages", None)
        image = kwargs.pop("image", None)
        
        if messages:
            messages.append({"role": "user", "content": prompt})
        else:
            messages = [{"role": "user", "content": prompt}]
            
        response = self.client.chat.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            stop=stop[0] if stop else None,
            image=image,
            **kwargs
        )
        
        if "error" in response:
            raise ValueError(f"API调用错误: {response['error']}")
            
        if "choices" not in response:
            raise ValueError("API响应格式错误")
        
        result = response["choices"][0]["message"]["content"]
        logger.info(f"LLM Response: {result}")
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
    logger.info("测试千帆大模型")
    llm = QianFanLLM()
    
    # 测试基本调用
    logger.info("测试基本调用：")
    logger.info(llm("你好，我是千帆大模型"))
    
    # 测试带历史消息的调用
    logger.info("\n测试带历史消息的调用：")
    history = [
        {"role": "user", "content": "你是谁？"},
        {"role": "assistant", "content": "我是千帆大模型。"}
    ]
    response = llm("我们之前聊了什么？", messages=history)
    logger.info(f"带历史响应：{response}")