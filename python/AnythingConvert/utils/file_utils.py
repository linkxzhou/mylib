"""
文件处理工具函数
"""

from pathlib import Path
from typing import List, Set
import mimetypes

def get_files_by_extension(directory: Path, extensions: List[str]) -> List[Path]:
    """
    根据扩展名获取目录下的所有文件
    
    Args:
        directory: 目录路径
        extensions: 文件扩展名列表
        
    Returns:
        List[Path]: 符合条件的文件路径列表
    """
    if not directory.exists() or not directory.is_dir():
        return []
    
    # 标准化扩展名
    ext_set = {ext.lower().lstrip('.') for ext in extensions}
    
    files = []
    for file_path in directory.rglob('*'):
        if file_path.is_file():
            file_ext = file_path.suffix.lower().lstrip('.')
            if file_ext in ext_set:
                files.append(file_path)
    
    return sorted(files)

def validate_output_path(output_path: Path, create_dirs: bool = True) -> bool:
    """
    验证输出路径是否有效
    
    Args:
        output_path: 输出路径
        create_dirs: 是否创建不存在的目录
        
    Returns:
        bool: 路径是否有效
    """
    try:
        if create_dirs:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 检查父目录是否存在且可写
        return output_path.parent.exists() and output_path.parent.is_dir()
        
    except Exception:
        return False

def get_file_size_mb(file_path: Path) -> float:
    """获取文件大小(MB)"""
    if not file_path.exists():
        return 0.0
    return file_path.stat().st_size / (1024 * 1024)

def detect_file_type(file_path: Path) -> str:
    """检测文件类型"""
    mime_type, _ = mimetypes.guess_type(str(file_path))
    if mime_type:
        return mime_type.split('/')[0]  # 返回主类型，如 'image', 'video', 'audio'
    return 'unknown'