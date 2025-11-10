import re
import json
import mmap
import gc
import numpy as np
from pathlib import Path
from typing import List, Optional, Iterator, Generator, Dict

punctuation = set("!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~.,;《》？！“”‘’@#￥%…&×（）——+【】{};；●，。&～、|\s:：\n")
en_punctuation = ",().!;:"
zh_punctuation = "，（）。！；："

def delete_file(file: str)-> bool:
    '''
    询问删除文件
    '''
    if exists(file):
        ans = input('delete file: {} ? Yes (y) or No (n)'.format(file))
        ans = ans.lower()
        if ans in ('yes', 'y'):
            remove(file)
            print('deleted.')
            return True
    return False

def remove_duplicate_punctuation(sentence: str) -> str:
    '''
    删除句子中重复的标点符号、重复的空格，同时将换行变为特殊字符'\n'
    '''
    # 将空格（全角空格）替换为逗号, 可能会有重复的空客，下面删除重复标点会删除
    sentence = re.sub(' |　', '，', sentence) 

    ans = ''
    n = len(sentence)
    p = 0
    while p < n:
        ans += sentence[p]

        while p + 1 < n and sentence[p] in punctuation and sentence[p + 1] in punctuation:
            p += 1
        p += 1

    return ans

def convert_en_punctuation_to_zh_punct(sentence: str) -> str:
    '''
    将句子中的英文标点替换文中文标点
    '''
    n = len(zh_punctuation)
    for i in range(n):
        sentence = sentence.replace(en_punctuation[i], zh_punctuation[i])
    return sentence

def get_sentences_dice_similarity(st_a: str, st_b: str) -> float:
    '''
    获取两个句子的Dice相似度（Dice similarity）
    s(a, b) =  2 * len( set(a) & set(b) ) / (len(set(a)) + len(set(b)))
    '''
    set_a, set_b = set(st_a), set(st_b)
    total_len  = len(set_a) + len(set_b)
    
    if total_len == 0: return 0.0

    inter_set =  set_a & set_b
    
    return ( 2 * len(inter_set)) / total_len

def read_texts_from_jsonl(file_path: Path, content_field: List[str], max_lines: Optional[int] = None) -> Iterator[str]:
    """从JSONL文件读取文本
    
    Args:
        file_path: JSONL文件路径
        max_lines: 最大读取行数
        
    Yields:
        str: 文本内容
    """
    count = 0
    try:
        with file_path.open('r', encoding='utf-8') as f:
            for line in f:
                if max_lines and count >= max_lines:
                    break
                try:
                    data = json.loads(line.strip())
                    # 尝试多个可能的文本字段
                    text = None
                    for field in content_field:
                        if field in data and data[field]:
                            text = remove_duplicate_punctuation(str(data[field]).strip())
                            break
                    
                    if text:
                        yield text
                        count += 1
                except json.JSONDecodeError as e:
                    continue
    except Exception as e:
        raise e

def read_texts_from_json(file_path: Path, content_field: List[str], max_lines: Optional[int] = None) -> Iterator[str]:
    """从JSON文件读取文本
    
    Args:
        file_path: JSON文件路径
        max_lines: 最大读取行数
        
    Yields:
        str: 文本内容
    """
    count = 0
    try:
        with file_path.open('r', encoding='utf-8') as f:
            data = json.load(f)
            
            # 如果是列表格式
            if isinstance(data, list):
                for item in data:
                    if max_lines and count >= max_lines:
                        break
                    
                    if isinstance(item, dict):
                        # 尝试多个可能的文本字段
                        text = None
                        for field in content_field:
                            if field in item and item[field]:
                                text = remove_duplicate_punctuation(str(item[field]).strip())
                                break
                        
                        if text:
                            yield text
                            count += 1
                    elif isinstance(item, str):
                        # 如果直接是字符串
                        text = remove_duplicate_punctuation(str(item).strip())
                        if text:
                            yield text
                            count += 1
            
            # 如果是字典格式
            elif isinstance(data, dict):
                # 尝试多个可能的文本字段
                text = None
                for field in content_field:
                    if field in data and data[field]:
                        if isinstance(data[field], list):
                            # 如果字段值是列表
                            for item in data[field]:
                                if max_lines and count >= max_lines:
                                    break
                                text = remove_duplicate_punctuation(str(item).strip())
                                if text:
                                    yield text
                                    count += 1
                        else:
                            # 如果字段值是字符串
                            text = remove_duplicate_punctuation(str(data[field]).strip())
                            if text:
                                yield text
                                count += 1
                        break
                        
    except Exception as e:
        raise e

def read_texts_from_txt(file_path: Path, max_lines: Optional[int] = None) -> Iterator[str]:
    """从TXT文件读取文本
    
    Args:
        file_path: TXT文件路径
        max_lines: 最大读取行数
        
    Yields:
        str: 文本内容
    """
    count = 0
    try:
        with file_path.open('r', encoding='utf-8') as f:
            for line in f:
                if max_lines and count >= max_lines:
                    break
                text = remove_duplicate_punctuation(line.strip())
                if text:  # 跳过空行
                    yield text
                    count += 1
    except Exception as e:
        raise e

def read_texts_from_parquet(file_path: Path, content_field: List[str], max_lines: Optional[int] = None) -> Iterator[str]:
    """从Parquet文件读取文本
    
    Args:
        file_path: Parquet文件路径
        max_lines: 最大读取行数
        
    Yields:
        str: 文本内容
    """
    count = 0
    try:
        # 读取 parquet 文件
        df = pd.read_parquet(file_path)
        
        # 遍历每一行
        for _, row in df.iterrows():
            if max_lines and count >= max_lines:
                break
            
            # 尝试多个可能的文本字段
            text = None
            for field in content_field:
                if field in row and pd.notna(row[field]) and row[field]:
                    text = remove_duplicate_punctuation(str(row[field]).strip())
                    break
            
            if text:
                yield text
                count += 1
                
    except Exception as e:
        raise e

def read_texts_from_jsonl_stream(file_path: Path) -> Generator[Dict, None, None]:
        """流式读取JSONL文件，避免一次性加载全部数据"""
        try:
            with file_path.open('rb') as f:
                # 使用内存映射读取大文件
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    for line in iter(mm.readline, b''):
                        if line.strip():
                            try:
                                yield json.loads(line.decode('utf-8'))
                            except Exception as e:
                                raise e
        except Exception as e:
            raise e

def split_text(text: str, n: int = 512) -> List[str]:
    """将文本分割成固定长度的片段"""
    return [text[i: i + n] for i in range(0, len(text), n)]

def filter_text_quality(text_input: any, content_field: List[str], min_length: int = 10, max_length: int = 4096) -> str:
    """根据配置过滤文本质量"""
    text = text_input
    if isinstance(text, dict):
        for col_name in content_field:
            if col_name in text_input:
                text = text_input[col_name]
                break
        
    if not text or not text.strip():
        return ""
    
    # 长度过滤
    if len(text) < min_length:
        return ""
    if len(text) > max_length:
        return ""
        
    # 简单的质量评估（可以根据需要扩展）
    # 检查是否包含过多的特殊字符
    special_char_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text)
    if special_char_ratio > 0.3:  # 特殊字符超过30%
        return ""
        
    # 检查重复字符
    if len(set(text.lower())) / len(text) < 0.1:  # 字符多样性太低
        return ""
        
    return remove_duplicate_punctuation(text)

def save_npfiletxt(file_path: Path, input_ids: List[int], nptype: np.dtype = np.uint16) -> None:
        """保存数组到文本文件"""
        try:
            with file_path.open('a') as f:
                np.savetxt(f, np.array(input_ids, nptype), fmt='%d', delimiter=',')
        except Exception as e:
            raise e

def save_binary_chunks(file_path: Path, input_ids: List[int], 
                        chunk_size: int = 1000000, nptype: np.dtype = np.uint16) -> None:
    """分块保存二进制数据，减少内存占用"""
    try:
        with file_path.open('ab') as f:
            for i in range(0, len(input_ids), chunk_size):
                chunk = input_ids[i:i + chunk_size]
                np.array(chunk, dtype=nptype).tofile(f)
                # 强制释放内存
                del chunk
                gc.collect()
    except Exception as e:
        raise e