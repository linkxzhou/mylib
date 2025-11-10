import base64
import io
from typing import Optional
import pickle
from pathlib import Path

def split_model_name(model_name: str) -> tuple[Optional[str], str]:
    """将模型名称拆分为组织名和模型名
    
    Args:
        model_name: 完整的模型名称，格式可以是 "org/model" 或 "model"
        
    Returns:
        tuple: 返回一个元组 (org_name, model_name)
            - 如果输入包含 "/"，返回 (组织名, 模型名)
            - 如果输入不包含 "/"，返回 (None, 模型名)
            
    Examples:
        >>> split_model_name("huggingface/bert-base")
        ("huggingface", "bert-base")
        >>> split_model_name("gpt-3")
        (None, "gpt-3")
    """
    if not isinstance(model_name, str):
        raise TypeError("模型名称必须是字符串类型")
        
    model_name = model_name.strip()
    if not model_name:
        raise ValueError("模型名称不能为空")
    
    # 特殊处理 huggingface
    if model_name.lower().startswith("huggingface"):
        return "huggingface", model_name[len("huggingface"):].strip("/")
        
    # 处理普通模型名称
    parts = [p.strip() for p in model_name.split("/") if p.strip()]
    
    if not parts:
        raise ValueError("无效的模型名称格式")
    elif len(parts) == 1:
        return None, parts[0]
    elif len(parts) == 2:
        return parts[0], parts[1]
    else:
        # 如果有多个斜杠，取第一个作为组织名，其余部分作为模型名
        return parts[0], "/".join(parts[1:])

def image_to_base64(image) -> Optional[str]:
    if image is None:
        return None

    if isinstance(image, str):  # 如果是文件路径
        with open(image, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
            
    # 如果是PIL Image
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def find_last_digit(self, string):
    for char in reversed(string):
        if char.isdigit():
            return char
            
    raise ValueError("No digit found in the string")

# 字符串转换 md5
def str_to_md5(src: str) -> str:
    import hashlib
    m = hashlib.md5()
    m.update(src.encode("utf-8"))
    return m.hexdigest()

def load_pkl(persistent_path: str) -> None:
    """
    从pickle文件中加载Python对象。
    :param persistent_path: 要加载的文件路径
    :return: 加载的Python对象，如果文件不存在则返回None
    """
    pickle_path = Path(persistent_path).with_suffix('.pkl')
    if pickle_path.exists():
        try:
            with open(pickle_path, 'rb') as f:
                python_object = pickle.load(f)
            return python_object
        except Exception as e:
            return None
    else:
        return None

def save_pkl(python_object: any, persistent_path: str) -> None:
    """
    将Python对象保存为pickle文件。
    :param python_object: 要保存的Python对象
    :param persistent_path: 保存的文件路径
    """
    pickle_path = Path(persistent_path).with_suffix('.pkl')
    with open(pickle_path, 'wb') as f:
        pickle.dump(python_object, f)