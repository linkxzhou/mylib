import time
from typing import Any, Dict, Optional
from threading import Lock

class Cache:
    """通用缓存存储类"""
    
    def __init__(self, ttl: int = 3600):
        """初始化缓存
        
        Args:
            ttl: 缓存过期时间（秒），默认1小时
        """
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()
        self._default_ttl = ttl
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """设置缓存
        
        Args:
            key: 缓存键
            value: 缓存值
            ttl: 过期时间（秒），None则使用默认值
        """
        with self._lock:
            self._cache[key] = {
                'value': value,
                'expire_at': time.time() + (ttl if ttl is not None else self._default_ttl)
            }
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取缓存值
        
        Args:
            key: 缓存键
            default: 默认值，当缓存不存在或已过期时返回
            
        Returns:
            缓存值或默认值
        """
        with self._lock:
            if key not in self._cache:
                return default
            
            cache_data = self._cache[key]
            if time.time() > cache_data['expire_at']:
                del self._cache[key]
                return default
                
            return cache_data['value']
    
    def delete(self, key: str) -> None:
        """删除缓存
        
        Args:
            key: 缓存键
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
    
    def clear(self) -> None:
        """清空所有缓存"""
        with self._lock:
            self._cache.clear()
    
    def cleanup(self) -> None:
        """清理过期缓存"""
        with self._lock:
            current_time = time.time()
            expired_keys = [
                key for key, data in self._cache.items()
                if current_time > data['expire_at']
            ]
            for key in expired_keys:
                del self._cache[key]
    
    def has_key(self, key: str) -> bool:
        """检查键是否存在且未过期
        
        Args:
            key: 缓存键
            
        Returns:
            bool: 是否存在且未过期
        """
        with self._lock:
            if key not in self._cache:
                return False
            
            if time.time() > self._cache[key]['expire_at']:
                del self._cache[key]
                return False
                
            return True

# 创建全局缓存实例
global_cache = Cache()

if __name__ == "__main__":
    # 测试代码
    cache = Cache(ttl=5)  # 5秒过期
    
    # 设置缓存
    cache.set("test_key", "test_value")
    print(cache.get("test_key"))  # 输出: test_value
    
    # 等待缓存过期
    time.sleep(6)
    print(cache.get("test_key"))  # 输出: None
    
    # 使用自定义过期时间
    cache.set("custom_ttl", "custom_value", ttl=2)
    print(cache.get("custom_ttl"))  # 输出: custom_value
    
    time.sleep(3)
    print(cache.get("custom_ttl"))  # 输出: None