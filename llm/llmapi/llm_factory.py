import ast
from typing import Optional, Dict, Any, List, Tuple
from langchain.llms.base import LLM
from llmapi.llm_plugin import Plugin, ReplaceHtmlPlugin, ReplaceImagePlugin
from llmapi.get_model_list import SUPPORTED_MODELS
from util.mylog import logger

class LLMChatAdapter:
    """适配器类，将LLM对象包装成具有appendSystemInfo和chat方法的接口"""
    
    def __init__(self, llm: LLM) -> None:
        self.llm = llm
        self.system_info = ""
        try:
            self.plugins: List[Plugin] = [ReplaceHtmlPlugin(), ReplaceImagePlugin()]
        except Exception as e:
            logger.warning(f"插件初始化失败: {str(e)}")
            self.plugins: List[Plugin] = []
        
    def appendSystemInfo(self, system_info: str) -> None:
        """添加系统提示信息"""
        self.system_info = system_info
        
    def chat(self, prompt: str, image: str = None) -> Tuple[bool, str]:
        """
        与LLM进行对话
        
        Args:
            prompt: 用户输入的提示
            
        Returns:
            Tuple[bool, str]: (是否成功, 响应内容或错误信息)
        """
        try:
            for plugin in self.plugins:
                prompt = plugin.process_input(prompt)
        except Exception as e:
            logger.warning(f"插件处理输入失败: {str(e)}")
            
        success = True
        response = None
        try:
            full_prompt = f"{self.system_info}\n\n{prompt}" if self.system_info else prompt
            response = self.llm(full_prompt, image=image)
        except Exception as e:
            success = False
            response = f"LLM调用出错: {str(e)}"
            logger.error(response)
            
        if success:
            try:
                for plugin in self.plugins:
                    response = plugin.process_output(response)
            except Exception as e:
                logger.warning(f"插件处理输出失败: {str(e)}")
                
        return success, response
    
    def literal_eval(self, response, replace_list: List[str] = None) -> Any:
        """
        安全地将LLM响应解析为Python对象
        
        Args:
            response: LLM响应，可以是字符串或(bool, str)元组
            
        Returns:
            解析后的Python对象
        """
        # 处理元组类型的响应
        if isinstance(response, tuple) and len(response) == 2:
            success, text = response
            if not success:
                logger.warning(f"LLM响应失败: {text}")
                return []
        else:
            text = response
        
        if replace_list:
            for replace_str in replace_list:
                text = text.replace(replace_str, "")

        try:
            # 尝试直接解析
            return ast.literal_eval(str(text).strip())
        except (ValueError, SyntaxError):
            # 如果直接解析失败，尝试提取可能的列表部分
            try:
                # 查找第一个 [ 和最后一个 ] 之间的内容
                text_str = str(text)
                start = text_str.find('[')
                end = text_str.rfind(']')
                if start != -1 and end != -1:
                    list_str = text_str[start:end + 1]
                    return ast.literal_eval(list_str)
                return []  # 如果没有找到列表，返回空列表
            except Exception as e:
                logger.warning(f"解析文本为Python对象失败: {str(e)}")
                return []  # 解析失败时返回空列表
            
class LLMFactory:
    """LLM工厂类，用于创建不同类型的LLM实例"""
    
    @classmethod
    def create(
        cls,
        model_type: str,
        model_name: Optional[str] = None,
        temperature: Optional[float] = 0.6,
        top_p: Optional[float] = 0.9,
        max_tokens: Optional[int] = None,
        **kwargs: Dict[str, Any]
    ) -> LLM:
        """
        创建LLM实例
        
        Args:
            model_type: 模型类型，支持 'hunyuan' 和 'qianfan'
            model_name: 模型名称
            temperature: 温度参数 (0.0-1.0)
            top_p: 采样参数 (0.0-1.0)
            max_tokens: 最大生成长度
            **kwargs: 其他参数
            
        Returns:
            LLM实例
            
        Raises:
            ValueError: 当模型类型不支持或参数无效时抛出
        """
        # 参数验证
        if temperature is not None and not 0 <= temperature <= 1:
            raise ValueError("temperature 必须在 0 到 1 之间")
        if top_p is not None and not 0 <= top_p <= 1:
            raise ValueError("top_p 必须在 0 到 1 之间")
        if max_tokens is not None and max_tokens <= 0:
            raise ValueError("max_tokens 必须大于 0")
            
        # 获取模型类
        model_type = model_type or "hunyuan"  # 默认使用 hunyuan
        model_class = SUPPORTED_MODELS.get(model_type)
        if not model_class:
            raise ValueError(f"不支持的模型类型: {model_type}，支持的类型: {list(SUPPORTED_MODELS.keys())}")
        
        logger.info(f"====== 使用 {model_type} 模型: {model_name}")
        # 创建实例
        return model_class(
            model_name=model_name,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            **kwargs
        )

if __name__ == "__main__":
    # 测试不同类型的模型
    test_prompts = [
        ("qianfan", "deepseek-v3"),
        ("hunyuan", "hunyuan-turbo"),
        ("huggingface", "Qwen/Qwen2.5-0.5B-Instruct"),
        ("openai", "gpt-3.5-turbo"),
        ("qwen", "qwen/qwen-plus"),
        ("zhipu", "glm-4")
    ]
    
    for model_type, model_name in test_prompts:
        try:
            logger.info(f"\n测试 {model_type} 模型: {model_name}")
            llm = LLMFactory.create(
                model_type=model_type,
                model_name=model_name,
                temperature=0.6
            )
            llm_adapter = LLMChatAdapter(llm)
            isok, response = llm_adapter.chat("你好，请介绍一下你自己")
            logger.info(f"响应: {response}, isok: {isok}")
        except Exception as e:
            logger.error(f"{model_type} 模型测试失败: {str(e)}")