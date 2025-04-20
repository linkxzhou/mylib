import os
import time
import json
import hmac
import base64
import hashlib
from typing import Dict, Any
from util.mylog import logger

class ZhipuBaseAPI:
    def __init__(self):
        self.api_key = os.environ.get("ZHIPU_API_KEY", "")
        if not self.api_key:
            logger.warning("未设置ZHIPU_API_KEY环境变量，请确保已正确配置API密钥")
        
        # 设置请求头
        self.headers = self._generate_headers()
    
    def _generate_headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }