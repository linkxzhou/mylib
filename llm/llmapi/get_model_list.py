from typing import Dict, Any, List
from qianfan.qianfan_llm import QianFanLLM
from qwen.qwen_llm import QwenLLM
from zhipu.zhipu_llm import ZhipuLLM
from myopenai.openai_llm import OpenAILLM
from myollama.ollama_llm import OllamaLLM
from siliconflow.siliconflow_llm import SiliconFlowLLM
from util.mylog import logger

SUPPORTED_MODELS = {
    "qianfan": QianFanLLM,
    "qwen": QwenLLM,
    "zhipu": ZhipuLLM,
    "openai": OpenAILLM,
    "ollama": OllamaLLM,
    "siliconflow": SiliconFlowLLM,
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