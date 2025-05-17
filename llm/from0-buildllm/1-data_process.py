import ujson
import numpy as np
import jsonlines
import time
import logging
import mmap
import gc
from pathlib import Path
from typing import List, Dict, Any, Iterator, Optional, Generator
from transformers import AutoTokenizer
from tqdm import tqdm
from itertools import islice

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class DataProcessor:
    def __init__(self, tokenizer_path: str = './my_tokenizer', base_path: str = "../datasets"):
        """初始化数据处理器"""
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
        self.base_path = Path(base_path)
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        # 添加批处理参数
        self.chunk_size = 10000  # 每次处理的数据量
        self.buffer_size = 1000000  # 缓冲区大小
        
    def save_npfiletxt(self, file_path: Path, input_ids: List[int], nptype: np.dtype = np.uint16) -> None:
        """保存数组到文本文件"""
        try:
            with file_path.open('a') as f:
                np.savetxt(f, np.array(input_ids, nptype), fmt='%d', delimiter=',')
        except Exception as e:
            logging.error(f"保存文件失败: {e}")
            raise

    def save_binary_chunks(self, file_path: Path, input_ids: List[int], 
                          chunk_size: int = 1000000, nptype: np.dtype = np.uint16) -> None:
        """分块保存二进制数据，减少内存占用"""
        try:
            with file_path.open('ab') as f:
                for i in range(0, len(input_ids), chunk_size):
                    chunk = input_ids[i:i + chunk_size]
                    np.array(chunk, nptype).tofile(f)
                    # 强制释放内存
                    del chunk
                    gc.collect()
        except Exception as e:
            logging.error(f"保存二进制文件失败: {e}")
            raise

    @staticmethod
    def split_text(text: str, n: int = 512) -> List[str]:
        """将文本分割成固定长度的片段"""
        return [text[i: i + n] for i in range(0, len(text), n)]

    def read_jsonl_stream(self, file_path: Path) -> Generator[Dict, None, None]:
        """流式读取JSONL文件，避免一次性加载全部数据"""
        try:
            with file_path.open('rb') as f:
                # 使用内存映射读取大文件
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    for line in iter(mm.readline, b''):
                        if line.strip():
                            try:
                                yield ujson.loads(line.decode('utf-8'))
                            except Exception as e:
                                logging.warning(f"解析JSONL行失败: {e}")
        except Exception as e:
            logging.error(f"读取JSONL文件失败: {file_path}, 错误: {e}")
            raise

    def process_wiki(self, data_path_list: List[Dict[str, str]]) -> None:
        """处理维基百科数据"""
        for file_info in data_path_list:
            file_path = Path(file_info["name"])
            file_type = file_info["type"]
            col_name = file_info["text"]
            
            try:
                logging.info(f"开始处理文件: {file_path}")
                output_path = file_path.with_suffix('.bin')
                
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
                            text_input = line[col_name]
                            text_arr = self.split_text(text_input)
                            for text in text_arr:
                                text_id = self.tokenizer(
                                    f'{self.bos_token}{text}{self.eos_token}'
                                ).data['input_ids']
                                if len(text_id) > 5:
                                    buffer.extend(text_id)
                                    
                            # 当缓冲区达到一定大小时写入文件
                            if len(buffer) >= self.buffer_size:
                                self.save_binary_chunks(output_path, buffer)
                                total_count += len(buffer)
                                logging.info(f"已处理 {total_count} 个token")
                                buffer = []
                                gc.collect()  # 强制垃圾回收
                                
                        except KeyError as e:
                            logging.warning(f"处理行时出错: {e}")
                            continue
                    
                    # 处理剩余数据
                    if buffer:
                        self.save_binary_chunks(output_path, buffer)
                        total_count += len(buffer)
                        logging.info(f"已处理 {total_count} 个token")
                        buffer = []
                        gc.collect()
                
                elif file_type == "json":
                    # 对于JSON文件，尝试分块处理
                    with file_path.open('r', encoding='utf-8') as file:
                        # 读取JSON数据
                        data = ujson.loads(file.read())
                    
                    # 分批处理数据
                    total_count = 0
                    for i in range(0, len(data), self.chunk_size):
                        chunk = data[i:i + self.chunk_size]
                        buffer = []
                        
                        for line in tqdm(chunk, desc=f"处理JSON块 {i//self.chunk_size + 1}/{(len(data)-1)//self.chunk_size + 1}"):
                            try:
                                text_input = line[col_name]
                                text_arr = self.split_text(text_input)
                                for text in text_arr:
                                    text_id = self.tokenizer(
                                        f'{self.bos_token}{text}{self.eos_token}'
                                    ).data['input_ids']
                                    if len(text_id) > 5:
                                        buffer.extend(text_id)
                            except KeyError as e:
                                logging.warning(f"处理JSON行时出错: {e}")
                                continue
                        
                        # 保存当前批次数据
                        if buffer:
                            self.save_binary_chunks(output_path, buffer)
                            total_count += len(buffer)
                            logging.info(f"已处理 {total_count} 个token")
                            buffer = []
                            # 释放内存
                            del chunk
                            gc.collect()
                
                else:
                    raise ValueError(f"不支持的文件类型: {file_type}")

                logging.info(f"已保存处理结果到: {output_path}")

            except Exception as e:
                logging.error(f"处理文件 {file_path} 时出错: {e}")
                raise

    def pretrain_process(self) -> None:
        """预训练数据处理"""
        try:
            data_path_list = [
                {
                    "name": str(self.base_path / "wikipedia-cn-20230720-filtered.json"),
                    "text": "completion",
                    "type": "json",
                },
                {
                    "name": str(self.base_path / "wikipedia-zh-cn-20241020.json"),
                    "text": "text",
                    "type": "jsonl",
                },
            ]
            
            # 处理各个数据文件
            self.process_wiki(data_path_list)
            
            # 流式合并处理结果
            output_path = self.base_path / "pretrain_data.csv"
            if output_path.exists():
                output_path.unlink()
                
            # 分块读取并写入
            with output_path.open('ab') as outfile:
                for file_info in data_path_list:
                    bin_path = Path(file_info["name"]).with_suffix('.bin')
                    logging.info(f"合并文件: {bin_path}")
                    
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
            
            # 获取文件大小信息
            file_size = output_path.stat().st_size
            num_elements = file_size // np.dtype(np.uint16).itemsize
            logging.info(f"预训练数据处理完成，文件大小: {file_size/1024/1024:.2f}MB, 约 {num_elements} 个token")
            
        except Exception as e:
            logging.error(f"预训练数据处理失败: {e}")
            raise

    def process_sft_batch(self, batch: List[Dict], max_length: int = 1024, 
                         padding: int = 0) -> List[int]:
        """处理SFT数据批次"""
        doc_ids = []
        
        for item in tqdm(batch, desc="处理SFT数据批次"):
            try:
                history, q, a = item['history'], item['q'], item['a']
                if len(q) < 10 or len(a) < 5:
                    continue

                messages = []
                for history_message in history:
                    if len(history_message) <= 1:
                        continue
                    messages.extend([
                        {"role": 'user', "content": history_message[0][:max_length]},
                        {"role": 'assistant', "content": history_message[1][:max_length]}
                    ])

                messages.extend([
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a},
                ])
                
                new_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                input_id = self.tokenizer(new_prompt).data['input_ids'][:max_length]
                input_id.extend([padding] * (max_length - len(input_id)))
                
                if len(input_id) >= 5:
                    doc_ids.extend(input_id)
                    
            except Exception as e:
                logging.warning(f"处理SFT数据项时出错: {e}")
                continue

        return doc_ids

    def sft_process(self) -> None:
        """SFT数据处理主函数"""
        try:
            output_path = self.base_path / "sft_data.csv"
            if output_path.exists():
                output_path.unlink()

            batch_size = 1000  # 减小批处理大小，降低内存占用
            sft_datasets = [
                self.base_path / "sft_data_zh.jsonl",
                # self.base_path / "sft_data_en.jsonl",
            ]

            for dataset_path in sft_datasets:
                logging.info(f"开始处理SFT数据集: {dataset_path}")
                total_processed = 0
                
                # 使用流式处理JSONL
                with jsonlines.open(dataset_path) as reader:
                    batch_num = 0
                    while True:
                        # 读取一批数据
                        batch = []
                        for _ in range(batch_size):
                            try:
                                obj = next(reader, None)
                                if obj is None:
                                    break
                                    
                                batch.append({
                                    'history': obj.get('history', ''),
                                    'q': obj.get('input', '') + obj.get('q', ''),
                                    'a': obj.get('output', '') + obj.get('a', '')
                                })
                            except jsonlines.InvalidLineError as e:
                                logging.warning(f"跳过无效的JSON行: {e}")
                                continue
                        
                        if not batch:
                            break
                            
                        # 处理批次
                        batch_num += 1
                        start_time = time.time()
                        doc_ids = self.process_sft_batch(batch, max_length=1024)
                        
                        # 保存批次结果
                        if doc_ids:
                            self.save_binary_chunks(output_path, doc_ids)
                            total_processed += len(doc_ids)
                            
                        process_time = time.time() - start_time
                        logging.info(f"完成第 {batch_num} 批处理，处理了 {len(doc_ids)} 个token，耗时: {process_time:.2f}秒")
                        
                        # 释放内存
                        del batch
                        del doc_ids
                        gc.collect()
                
                logging.info(f"完成 {dataset_path} 处理，总共处理了 {total_processed} 个token")

        except Exception as e:
            logging.error(f"SFT数据处理失败: {e}")
            raise

def main():
    """主函数"""
    try:
        # 设置较低的初始内存使用
        processor = DataProcessor()
        process_types = ['pretrain', 'sft']
        
        if 'pretrain' in process_types:
            processor.pretrain_process()
            # 处理完预训练数据后强制清理内存
            gc.collect()
            
        if 'sft' in process_types:
            processor.sft_process()
            
    except Exception as e:
        logging.error(f"程序执行失败: {e}")
        raise

if __name__ == "__main__":
    main()