from itertools import islice
from typing import List, Dict, Any
from duckduckgo_search import DDGS
from util.mylog import logger

class DuckDuckGoSearch:
    """DuckDuckGo 搜索功能封装"""
    
    def __init__(self):
        self.ddgs = DDGS()
    
    def search(self, keywords: str, max_results: int = 10, safesearch: str = 'Off', 
              timelimit: str = 'y') -> Dict[str, List[Dict[str, Any]]]:
        """通用文本搜索
        
        Args:
            keywords: 搜索关键词
            max_results: 最大结果数量
            safesearch: 安全搜索选项 ('On' or 'Off')
            timelimit: 时间限制 ('d', 'w', 'm', 'y')
            
        Returns:
            包含搜索结果的字典
        """
        try:
            results = []
            ddgs_gen = self.ddgs.text(
                keywords, 
                safesearch=safesearch,
                timelimit=timelimit,
                backend="lite"
            )
            for r in islice(ddgs_gen, max_results):
                results.append(r)
            return {'results': results}
        except Exception as e:
            logger.error(f"文本搜索出错: {str(e)}")
            return {'results': [], 'error': str(e)}

    def search_answers(self, keywords: str, max_results: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """问答搜索
        
        Args:
            keywords: 搜索关键词
            max_results: 最大结果数量
            
        Returns:
            包含答案的字典
        """
        try:
            results = []
            # 使用 text 方法替代 answers 方法
            ddgs_gen = self.ddgs.text(
                keywords,
                safesearch='Off',
                timelimit='y',
                backend="lite",
                region='wt-wt'  # 使用全球区域
            )
            for r in islice(ddgs_gen, max_results):
                results.append(r)
            return {'results': results}
        except Exception as e:
            logger.error(f"问答搜索出错: {str(e)}")
            return {'results': [], 'error': str(e)}

    def search_images(self, keywords: str, max_results: int = 10, 
                     safesearch: str = 'Off') -> Dict[str, List[Dict[str, Any]]]:
        """图片搜索
        
        Args:
            keywords: 搜索关键词
            max_results: 最大结果数量
            safesearch: 安全搜索选项 ('On' or 'Off')
            
        Returns:
            包含图片信息的字典
        """
        try:
            results = []
            ddgs_gen = self.ddgs.images(
                keywords,
                safesearch=safesearch,
                timelimit=None
            )
            for r in islice(ddgs_gen, max_results):
                results.append(r)
            return {'results': results}
        except Exception as e:
            logger.error(f"图片搜索出错: {str(e)}")
            return {'results': [], 'error': str(e)}

    def search_videos(self, keywords: str, max_results: int = 10, 
                     safesearch: str = 'Off', resolution: str = "high") -> Dict[str, List[Dict[str, Any]]]:
        """视频搜索
        
        Args:
            keywords: 搜索关键词
            max_results: 最大结果数量
            safesearch: 安全搜索选项 ('On' or 'Off')
            resolution: 视频分辨率 ("high" or "standard")
            
        Returns:
            包含视频信息的字典
        """
        try:
            results = []
            ddgs_gen = self.ddgs.videos(
                keywords,
                safesearch=safesearch,
                timelimit=None,
                resolution=resolution
            )
            for r in islice(ddgs_gen, max_results):
                results.append(r)
            return {'results': results}
        except Exception as e:
            logger.error(f"视频搜索出错: {str(e)}")
            return {'results': [], 'error': str(e)}

# Update the usage example
if __name__ == "__main__":
    ddg = DuckDuckGoSearch()
    
    # 文本搜索
    text_results = ddg.search("Python programming", max_results=5)
    logger.info("文本搜索结果:", text_results)
    
    # 问答搜索
    answer_results = ddg.search_answers("What is Python?", max_results=3)
    logger.info("问答搜索结果:", answer_results)
    
    # 图片搜索
    image_results = ddg.search_images("cute cats", max_results=5)
    logger.info("图片搜索结果:", image_results)
    
    # 视频搜索
    video_results = ddg.search_videos("Python tutorials", max_results=5)
    logger.info("视频搜索结果:", video_results)