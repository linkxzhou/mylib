from util.mylog import logger
import re
import base64
from pathlib import Path
import os

class Plugin:
    """LLM插件基类"""
    def process_input(self, prompt: str) -> str:
        logger.debug(f"Request: {prompt}")
        return prompt
    
    def process_output(self, response: str) -> str:
        logger.debug(f"Response: {response}")
        return response

class ReplaceHtmlPlugin(Plugin):
    """替换某些字符的的插件"""
    def __init__(self):
        self.replace_html = ['<', '>']
        
    def process_output(self, response: str) -> str:
        logger.debug(f"Response: {response}")
        for html in self.replace_html:
            response = response.replace(html, f"\{html}")
        return response

class ReplaceImagePlugin(Plugin):
    """替换图片的插件"""
    def __init__(self):
        # 修改正则表达式，避免使用可变长度的 look-behind
        self.pattern = r'(?:^|\s)([^\s()<>\[\]]+\.(?:png|jpg|jpeg|gif|bmp|webp))(?!\))'
        
    def convert_to_base64(self, image_path):
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode()
                ext = Path(image_path).suffix.lstrip('.')
                return f"data:image/{ext};base64,{encoded_string}"
        except Exception:
            return None
            
    def replace_image(self, match):
        path = match.group(1)
        if path.startswith(('http://', 'https://', 'data:')):
            return path
        
        abs_path = os.path.abspath(os.path.expanduser(path))
        if os.path.isfile(abs_path):
            base64_data = self.convert_to_base64(abs_path)
            if base64_data:
                return f"![image]({base64_data})"
        return path

    def process_output(self, response: str) -> str:
        logger.debug(f"Response: {response}")
        return re.sub(self.pattern, self.replace_image, response)
        