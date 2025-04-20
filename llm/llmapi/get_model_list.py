from typing import Optional, Dict, Any, List, Tuple
from llmapi.hunyuan.hunyuan_llm import HunYuanLLM
from llmapi.qianfan.qianfan_llm import QianFanLLM
from llmapi.qwen.qwen_llm import QwenLLM
from llmapi.zhipu.zhipu_llm import ZhipuLLM
from llmapi.myopenai.openai_llm import OpenAILLM
from llmapi.myollama.ollama_llm import OllamaLLM
from util.mylog import logger

SUPPORTED_MODELS = {
    "hunyuan": HunYuanLLM,
    "qianfan": QianFanLLM,
    "qwen": QwenLLM,
    "zhipu": ZhipuLLM,
    "openai": OpenAILLM,
    "ollama": OllamaLLM,
}

def get_text_model_list() -> List[Dict[str, Any]]:
    """
    获取可用的文本模型列表
    
    Returns:
        List[Dict[str, Any]]: 模型信息列表，每个模型包含 name 和 description
    """
    model_list = []
    for api_name, api_class in SUPPORTED_MODELS.items():
        try:
            api = api_class()
            models = api.client.get_model_list()
            if models:
                model_list.extend(models)
                logger.info(f"======== 成功加载 {api_name} 模型列表: {len(models)} 个模型")
            else:
                logger.warning(f"======== {api_name} 模型列表为空")
        except Exception as e:
            logger.error(f"======== 加载 {api_name} 模型列表失败: {str(e)}")
            continue

    return model_list

def get_mutli_model_list() -> List[Dict[str, Any]]:
    """
    获取可用的多模态模型列表

    Returns:
        List[Dict[str, Any]]: 模型信息列表，每个模型包含 name 和 description
    """
    return []

if __name__ == "__main__":
    # 获取并显示模型列表
    model_list = get_text_model_list()
    logger.info(f"可用模型列表: {model_list}")