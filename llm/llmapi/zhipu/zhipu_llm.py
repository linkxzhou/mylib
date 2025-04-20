from util.mylog import logger
from typing import Any, List, Mapping, Optional, Dict
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from pydantic import Field
from zhipu.zhipu_text import ZhipuTextAPI

class ZhipuLLM(LLM):
    """智谱 AI 大模型的 LangChain LLM 实现"""
    
    client: ZhipuTextAPI = Field(default_factory=ZhipuTextAPI)
    model_name: str = Field(default="glm-4")
    temperature: Optional[float] = Field(default=0.6)
    top_p: Optional[float] = Field(default=0.9)
    max_tokens: Optional[int] = Field(default=None)
    
    class Config:
        arbitrary_types_allowed = True
    
    @property
    def _llm_type(self) -> str:
        """返回 LLM 类型"""
        return "zhipu"

    @property
    def _supported_params(self) -> List[str]:
        """Return supported parameters."""
        return ["temperature", "top_p", "max_tokens", "model"]

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling Zhipu API."""
        return {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 4096,
            "model": "glm-4"
        }
        
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """执行 LLM 调用，支持文本和图像输入"""
        image = kwargs.pop("image", None)
        
        if image is not None:
            response = self.client.generate_text_with_image(
                prompt=prompt,
                image=image,
                model="glm-4v",  # 使用多模态模型
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                **kwargs
            )
        else:
            response = self.client.generate_text(
                prompt=prompt,
                model=self.model_name,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                stop=stop[0] if stop else None,
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
    logger.info("测试智谱大模型")
    llm = ZhipuLLM()
    
    # 测试文本生成
    logger.info("测试文本生成：")
    logger.info(llm("你好，我是智谱大模型"))
    
    # 测试图文生成
    from PIL import Image
    logger.info("测试图文生成：")
    try:
        image_path = "test.jpg"  # 替换为实际的测试图片路径
        image = Image.open(image_path)
        response = llm("这张图片是什么内容？", image=image)
        logger.info(f"图文响应：{response}")
    except Exception as e:
        logger.error(f"图文生成测试失败：{str(e)}")