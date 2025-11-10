import logging
from typing import Optional

class logger:
    """日志工具类，提供静态方法用于记录不同级别的日志"""
    
    @staticmethod
    def __get_logger():
        return logging.getLogger(__name__)
    
    @staticmethod
    def info(message):
        logger.__get_logger().info(message)
    
    @staticmethod
    def debug(message):
        logger.__get_logger().debug(message)
    
    @staticmethod
    def error(message):
        logger.__get_logger().error(message)
    
    @staticmethod
    def warning(message):
        logger.__get_logger().warning(message)
    
    @staticmethod
    def critical(message):
        logger.__get_logger().critical(message)
        
def setup_logging(
    level=logging.INFO,
    log_file: Optional[str] = None,
    console: bool = True
):
    """设置日志系统
    
    Args:
        level: 日志级别
        log_file: 日志文件路径，None表示不记录到文件
        console: 是否输出到控制台
        queue_handler: 是否添加队列处理器
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # 清除现有的处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # 添加控制台处理器
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # 添加文件处理器
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

# 初始化日志系统
setup_logging(
    level=logging.INFO,
    console=True,
)