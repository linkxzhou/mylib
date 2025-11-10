import numpy as np
import json
import mmap
import gc
from pathlib import Path
from typing import List, Dict, Any, Generator
from transformers import AutoTokenizer
from tqdm import tqdm

try:
    from pre_configurator import llmconfig, logger
    logger.info("已加载pre_configurator配置")
except ImportError:
    print("未找到pre_configurator.py，将使用默认配置")
    raise e

class SFTDataProcessor:
    def __init__(self, cfg: Dict[str, Any] = None):
        """初始化数据处理器"""
        self.config = cfg or llmconfig
        
        # 从配置中获取路径和参数
        self.tokenizer_path = self.config.get('paths.tokenizer_dir', 'my_tokenizer')
        self.base_path = Path(self.config.get('paths.base_dir', 'datasets'))
        self.output_dir = Path(self.config.get('paths.output_dir', 'outputs'))
        self.cache_dir = Path(self.config.get('paths.cache_dir', 'cache'))
        
        # 确保目录存在
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        
        # 从配置中获取特殊token
        self.bos_token = self.config.get('tokenizer_config.bos_token', '<s>')
        self.eos_token = self.config.get('tokenizer_config.eos_token', '</s>')
        self.pad_token = self.config.get('tokenizer_config.pad_token', '<pad>')
        self.unk_token = self.config.get('tokenizer_config.unk_token', '<unk>')
        
        # 从配置中获取数据处理参数
        self.max_seq_length = self.config.get('dataset.max_seq_length', 2048)
        self.sft_preprocessing = self.config.get('dataset.sft_preprocessing', {})
        
        # 批处理参数
        self.chunk_size = self.config.get('dataset.chunk_size', 10000)  # 每次处理的数据量
        self.buffer_size = self.config.get('dataset.buffer_size', 1000000)  # 缓冲区大小
        
    def save_npfiletxt(self, file_path: Path, input_ids: List[int], nptype: np.dtype = np.uint16) -> None:
        """保存数组到文本文件"""
        try:
            with file_path.open('a') as f:
                np.savetxt(f, np.array(input_ids, nptype), fmt='%d', delimiter=',')
        except Exception as e:
            logger.error(f"保存文件失败: {e}")
            raise

    def save_binary_chunks(self, file_path: Path, input_ids: List[int], 
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
            logger.error(f"保存二进制文件失败: {e}")
            raise

    @staticmethod
    def split_text(text: str, n: int = 512) -> List[str]:
        """将文本分割成固定长度的片段"""
        return [text[i: i + n] for i in range(0, len(text), n)]
    
    def filter_text_quality(self, text: str) -> bool:
        """根据配置过滤文本质量"""
        if not text or not text.strip():
            return False
            
        # 长度过滤
        if len(text) < self.sft_preprocessing.get('min_length', 10):
            return False
        if len(text) > self.sft_preprocessing.get('max_length', 4096):
            return False
            
        # 简单的质量评估（可以根据需要扩展）
        # 检查是否包含过多的特殊字符
        special_char_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text)
        if special_char_ratio > 0.3:  # 特殊字符超过30%
            return False
            
        # 检查重复字符
        if len(set(text.lower())) / len(text) < 0.1:  # 字符多样性太低
            return False
            
        return True
    
    def remove_duplicates(self, texts: List[str]) -> List[str]:
        """移除重复文本"""
        if not self.sft_preprocessing.get('remove_duplicates', True):
            return texts
            
        seen = set()
        unique_texts = []
        for text in texts:
            text_hash = hash(text.strip().lower())
            if text_hash not in seen:
                seen.add(text_hash)
                unique_texts.append(text)
        
        logger.info(f"去重前: {len(texts)} 条，去重后: {len(unique_texts)} 条")
        return unique_texts

    def format_sft_data(self, data_item: Dict[str, str], col_names: List[str]) -> str:
        """将SFT数据格式化为训练文本"""
        formatted_parts = []
        
        # 按照 col_names 顺序拼接字段
        for col_name in col_names:
            if col_name in data_item:
                field_value = data_item[col_name].strip()
                if field_value:  # 只有非空字段才添加
                    formatted_parts.append(f"### {col_name}:\n{field_value}")
        
        # 用双换行符连接所有部分
        formatted_text = "\n\n".join(formatted_parts)
        
        if len(formatted_text) == 0:
            return None
        
        return formatted_text

    def read_jsonl_stream(self, file_path: Path) -> Generator[Dict, None, None]:
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
                                logger.warning(f"解析JSONL行失败: {e}")
        except Exception as e:
            logger.error(f"读取JSONL文件失败: {file_path}, 错误: {e}")
            raise

    def process_data(self, data_path_list: List[Dict[str, str]]) -> None:
        """处理数据"""
        for file_info in data_path_list:
            file_path = Path(file_info.get("name", ""))
            file_type = file_info.get("type", "")
            col_names = file_info.get("text", "")
            
            try:
                logger.info(f"开始处理文件: {file_path}")
                output_path = self.output_dir / f"{file_path.stem}.bin"
                
                # 清空输出文件
                if output_path.exists():
                    output_path.unlink()
                
                # 流式处理数据
                if file_type == "jsonl":
                    # 使用流式处理JSONL
                    data_iterator = self.read_jsonl_stream(file_path)
                    total_count = 0
                    buffer = []
                    
                    for line in tqdm(data_iterator, desc=f"处理 {file_path.name}"):
                        try:
                            # 对于SFT数据，使用专门的格式化方法
                            text_input = self.format_sft_data(line, col_names)
                            if not text_input:
                                continue
                            
                            # 应用质量过滤
                            if not self.filter_text_quality(text_input):
                                continue
                                
                            text_arr = self.split_text(text_input, n=self.max_seq_length//2)
                            for text in text_arr:
                                if self.filter_text_quality(text):
                                    text_id = self.tokenizer(
                                        f'{self.bos_token}{text}{self.eos_token}',
                                        max_length=self.max_seq_length,
                                        truncation=True
                                    )['input_ids']
                                    if len(text_id) > 5:
                                        buffer.extend(text_id)
                                    
                            # 当缓冲区达到一定大小时写入文件
                            if len(buffer) >= self.buffer_size:
                                self.save_binary_chunks(output_path, buffer)
                                total_count += len(buffer)
                                logger.info(f"已处理 {total_count} 个token")
                                buffer = []
                                gc.collect()  # 强制垃圾回收
                                
                        except KeyError as e:
                            logger.warning(f"处理行时出错: {e}")
                            continue
                    
                    # 处理剩余数据
                    if buffer:
                        self.save_binary_chunks(output_path, buffer)
                        total_count += len(buffer)
                        logger.info(f"已处理 {total_count} 个token")
                        buffer = []
                        gc.collect()
                
                elif file_type == "json":
                    # 对于JSON文件，尝试分块处理
                    with file_path.open('r', encoding='utf-8') as file:
                        # 读取JSON数据
                        data = json.loads(file.read())
                    
                    # 分批处理数据
                    total_count = 0
                    for i in range(0, len(data), self.chunk_size):
                        chunk = data[i:i + self.chunk_size]
                        buffer = []
                        
                        for line in tqdm(chunk, desc=f"处理JSON块 {i//self.chunk_size + 1}/{(len(data)-1)//self.chunk_size + 1}"):
                            try:
                                text_input = None
                                for col_name in col_names:
                                    if col_name in line:
                                        text_input = line[col_name]
                                        break
                                
                                # 应用质量过滤
                                if not self.filter_text_quality(text_input):
                                    continue
                                    
                                text_arr = self.split_text(text_input, n=self.max_seq_length//2)
                                for text in text_arr:
                                    if self.filter_text_quality(text):
                                        text_id = self.tokenizer(
                                            f'{self.bos_token}{text}{self.eos_token}',
                                            max_length=self.max_seq_length,
                                            truncation=True
                                        )['input_ids']
                                        if len(text_id) > 5:
                                            buffer.extend(text_id)
                            except KeyError as e:
                                logger.warning(f"处理JSON行时出错: {e}")
                                continue
                        
                        # 保存当前批次数据
                        if buffer:
                            self.save_binary_chunks(output_path, buffer)
                            total_count += len(buffer)
                            logger.info(f"已处理 {total_count} 个token")
                            buffer = []
                            # 释放内存
                            del chunk
                            gc.collect()
                
                else:
                    raise ValueError(f"不支持的文件类型: {file_type}")

                logger.info(f"已保存处理结果到: {output_path}")

            except Exception as e:
                logger.error(f"处理文件 {file_path} 时出错: {e}")
                raise

    def sft_process(self) -> None:
        """sft数据处理"""
        try:
            # 从配置中获取训练文件列表
            train_files = self.config.get('dataset.sft_preprocessing.files', [])
            content_field = self.config.get('dataset.sft_preprocessing.content_field', [])
            data_path_list = []

            # 根据文件扩展名确定类型和文本字段
            for file_name in train_files:
                logger.info(f"处理文件: {file_name}")
                file_path = Path(file_name)
                if not file_path.exists():    
                    logger.error(f"文件不存在: {file_path}")
                    continue

                if file_path.suffix == '.jsonl':
                    data_path_list.append({
                        "name": str(file_path),
                        "text": content_field,  # 默认文本字段
                        "type": "jsonl",
                    })
                elif file_path.suffix == '.json':
                    data_path_list.append({
                        "name": str(file_path),
                        "text": content_field,  # 默认文本字段
                        "type": "json",
                    })
                elif file_path.suffix == '.txt':
                    data_path_list.append({
                        "name": str(file_path),
                        "text": content_field,  # 文本文件字段
                        "type": "txt",
                    })
            
            if not data_path_list:
                logger.warning("未找到有效的训练文件")
                raise ValueError("未找到有效的训练文件")
            
            # 处理各个数据文件
            self.process_data(data_path_list)
            
            # 流式合并处理结果
            output_path = self.output_dir / "sft_data.bin"
            if output_path.exists():
                output_path.unlink()
                
            # 分块读取并写入
            with output_path.open('ab') as outfile:
                for file_info in data_path_list:
                    bin_path = self.output_dir / f"{Path(file_info.get('name', '')).stem}.bin"
                    if bin_path.exists():
                        logger.info(f"合并文件: {bin_path}")
                        
                        # 使用内存映射读取大文件
                        with bin_path.open('rb') as infile:
                            # 分块读取并写入
                            chunk_size = 10 * 1024 * 1024  # 10MB chunks
                            while True:
                                chunk = infile.read(chunk_size)
                                if not chunk:
                                    break
                                
                                outfile.write(chunk)
                                # 释放内存
                                del chunk
                                gc.collect()
                    else:
                        logger.warning(f"二进制文件不存在: {bin_path}")
            
            # 获取文件大小信息
            file_size = output_path.stat().st_size
            num_elements = file_size // np.dtype(np.uint16).itemsize
            logger.info(f"预训练数据处理完成，文件大小: {file_size/1024/1024:.2f}MB, 约 {num_elements} 个token")
            return True
            
        except Exception as e:
            logger.error(f"预训练数据处理失败: {e}")
            return False

def main():
    """主函数"""
    try:
        # 初始化数据处理器
        processor = DataProcessor(llmconfig)

        logger.info("开始sft数据处理...")
        success = processor.sft_process()
        if not success:
            logger.error("sft数据处理失败")
            return
            
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        raise

    # 处理完预训练数据后强制清理内存
    gc.collect()

if __name__ == "__main__":
    main()