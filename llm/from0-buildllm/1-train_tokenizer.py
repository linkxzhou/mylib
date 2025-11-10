#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tokenizer训练脚本 - 从零开始训练BPE Tokenizer

功能:
1. 从零开始训练BPE tokenizer
2. 支持JSONL和TXT格式的训练数据
3. 自动保存训练结果和配置
4. 提供评估功能
5. 兼容HuggingFace格式

使用方法:
    python 0-train_tokenizer.py
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
from pre_utils import read_texts_from_jsonl, read_texts_from_txt, read_texts_from_json

try:
    from transformers import PreTrainedTokenizerFast
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
except ImportError as e:
    print(f"请安装必要的库: pip install sentencepiece transformers tokenizers")
    raise e

try:
    from pre_configurator import llmconfig, logger
    logger.info("已加载pre_configurator配置")
except ImportError:
    print("未找到pre_configurator.py，将使用默认配置")
    raise e

@dataclass
class TokenizerConfig:
    # 基础配置
    tokenizer_type: str = llmconfig.get('tokenizer_config.model_type', "BPE")  # tokenizer类型: BPE, WordPiece, Unigram
    vocab_size: int = llmconfig.get('tokenizer_config.vocab_size', 10240)  # 词汇表大小
    model_max_length: int = llmconfig.get('tokenizer_config.model_max_length', 2048)  # 最大序列长度
    
    # 训练数据配置
    max_lines_per_file: Optional[int] = None  # 每个文件最大行数
    
    # BPE训练配置
    min_frequency: int = 2  # 最小频率
    show_progress: bool = True  # 显示训练进度
    
    # 特殊token配置
    unk_token: str = llmconfig.get('tokenizer_config.unk_token', "<unk>")  # 未知token
    bos_token: str = llmconfig.get('tokenizer_config.bos_token', "<s>")  # 开始token
    eos_token: str = llmconfig.get('tokenizer_config.eos_token', "</s>")  # 结束token
    pad_token: str = llmconfig.get('tokenizer_config.pad_token', "<pad>")  # 填充token
    
    # 对话特殊token
    special_tokens: List[str] = field(default_factory=lambda: llmconfig.get('tokenizer_config.additional_special_tokens', [
        "<|system|>", "<|user|>", "<|assistant|>",  # 对话角色token
        "<|im_start|>", "<|im_end|>",  # 消息边界token
    ]))
    
    # 输出配置
    output_dir: str = llmconfig.get('paths.tokenizer_dir', 'my_tokenizer') # 输出目录
    
    # 评估配置
    eval_sample_size: int = 1000  # 评估样本大小

class BPETokenizerTrainer:
    """从零开始的BPE Tokenizer训练器"""
    
    def __init__(self, config: Optional[TokenizerConfig] = None):
        """初始化训练器
        
        Args:
            config: tokenizer配置，如果为None则自动加载
        """
        self.config = config or TokenizerConfig()
        self.tokenizer_dir = Path(self.config.output_dir)
        self.tokenizer = None
        self.files = llmconfig.get('dataset.tokenizer.files', [])
        self.content_field = llmconfig.get('dataset.tokenizer.content_field', [])
        
        logger.info(f"初始化BPETokenizerTrainer")
        logger.info(f"Tokenizer类型: {self.config.tokenizer_type}")
        logger.info(f"目标词汇表大小: {self.config.vocab_size}")
        logger.info(f"训练数据目录: {self.files}")
        logger.info(f"输出目录: {self.tokenizer_dir}")
        logger.info(f"训练数据字段: {self.content_field}")

    def prepare_training_data(self) -> str:
        """准备训练数据，将所有文本合并到一个临时文件
        
        Returns:
            str: 临时训练文件路径
        """
        logger.info("开始准备训练数据...")
        self.content_field = llmconfig.get('dataset.tokenizer.content_field', [])
        # 创建临时训练文件
        temp_file = self.tokenizer_dir / "temp_training_data.txt"
        self.tokenizer_dir.mkdir(parents=True, exist_ok=True)
        
        max_lines = self.config.max_lines_per_file
        processed_texts = 0
        
        with temp_file.open('w', encoding='utf-8') as outf:
            # 处理每个文件
            for file_name in self.files:
                logger.info(f"处理文件: {file_name}")
                # 检查文件是否存在
                file_path = Path(file_name)
                if not file_path.exists():    
                    logger.error(f"文件不存在: {file_path}")
                    continue
                
                if file_path.suffix == '.jsonl':
                    text_iterator = read_texts_from_jsonl(file_path, self.content_field, max_lines)
                elif file_path.suffix == '.json':
                    text_iterator = read_texts_from_json(file_path, self.content_field, max_lines)
                elif file_path.suffix == '.txt':
                    text_iterator = read_texts_from_txt(file_path, max_lines)
                else:
                    logger.warning(f"不支持的文件格式: {file_path}")
                    continue
                    
                # 写入文本
                for text in text_iterator:
                    if text.strip():  # 跳过空行
                        outf.write(text.strip() + '\n')
                        processed_texts += 1
                        if processed_texts % 10000 == 0:
                            logger.info(f"已处理 {processed_texts} 条文本")
                    
        if processed_texts == 0:
            raise ValueError("没有有效文本可用于训练")
        
        logger.info(f"数据准备完成，共处理 {processed_texts} 条文本")
        logger.info(f"临时文件: {temp_file}")
        
        return str(temp_file)

    def train(self) -> bool:
        """训练tokenizer
        
        Returns:
            bool: 训练是否成功
        """
        start_time = time.time()
        
        try:
            logger.info(f"开始从零训练{self.config.tokenizer_type} tokenizer...")
            
            # 准备训练数据
            training_file = self.prepare_training_data()
            if not training_file:
                raise ValueError("没有有效文本可用于训练")
            
            # 创建tokenizer
            if self.config.tokenizer_type == "BPE":
                tokenizer = Tokenizer(models.BPE(unk_token=self.config.unk_token))
            elif self.config.tokenizer_type == "WordPiece":
                tokenizer = Tokenizer(models.WordPiece(unk_token=self.config.unk_token))
            elif self.config.tokenizer_type == "Unigram":
                tokenizer = Tokenizer(models.Unigram())
            else:
                raise ValueError(f"不支持的tokenizer类型: {self.config.tokenizer_type}")
            
            # 设置预处理器
            tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            
            # 设置解码器
            tokenizer.decoder = decoders.ByteLevel()
            
            # 准备特殊token列表
            special_tokens = [
                self.config.unk_token,
                self.config.bos_token, 
                self.config.eos_token,
                self.config.pad_token
            ] + self.config.special_tokens
            
            # 创建训练器
            if self.config.tokenizer_type == "BPE":
                trainer = trainers.BpeTrainer(
                    vocab_size=self.config.vocab_size,
                    min_frequency=self.config.min_frequency,
                    special_tokens=special_tokens,
                    show_progress=self.config.show_progress
                )
            elif self.config.tokenizer_type == "WordPiece":
                trainer = trainers.WordPieceTrainer(
                    vocab_size=self.config.vocab_size,
                    min_frequency=self.config.min_frequency,
                    special_tokens=special_tokens,
                    show_progress=self.config.show_progress
                )
            elif self.config.tokenizer_type == "Unigram":
                trainer = trainers.UnigramTrainer(
                    vocab_size=self.config.vocab_size,
                    special_tokens=special_tokens,
                    show_progress=self.config.show_progress
                )
            
            # 训练tokenizer
            logger.info(f"开始训练，目标词汇表大小: {self.config.vocab_size}")
            tokenizer.train([training_file], trainer)
            
            # 设置后处理器
            tokenizer.post_processor = processors.TemplateProcessing(
                single=f"{self.config.bos_token} $A {self.config.eos_token}",
                pair=f"{self.config.bos_token} $A {self.config.eos_token} $B:1 {self.config.eos_token}:1",
                special_tokens=[
                    (self.config.bos_token, tokenizer.token_to_id(self.config.bos_token)),
                    (self.config.eos_token, tokenizer.token_to_id(self.config.eos_token)),
                ]
            )
            
            # 创建HuggingFace兼容的tokenizer
            self.tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=tokenizer,
                unk_token=self.config.unk_token,
                bos_token=self.config.bos_token,
                eos_token=self.config.eos_token,
                pad_token=self.config.pad_token,
                model_max_length=self.config.model_max_length
            )
            
            # 验证词汇表大小
            final_vocab_size = len(self.tokenizer)
            logger.info(f"最终词汇表大小: {final_vocab_size}")
            
            # 保存tokenizer
            self._save_tokenizer()
            
            # 清理临时文件
            Path(training_file).unlink(missing_ok=True)
            
            total_time = time.time() - start_time
            logger.info(f"Tokenizer训练完成，总耗时: {total_time:.2f}秒")
            return True
            
        except Exception as e:
            logger.error(f"训练过程发生错误: {e}")
            return False

    def _save_tokenizer(self) -> None:
        """保存tokenizer和相关信息"""
        try:
            # 创建目录
            self.tokenizer_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"保存tokenizer到: {self.tokenizer_dir}")
            
            # 保存tokenizer
            self.tokenizer.save_pretrained(str(self.tokenizer_dir))
            logger.info("已保存tokenizer文件")
            
            # 保存训练信息
            training_info = {
                "tokenizer_type": self.config.tokenizer_type,
                "vocab_size": len(self.tokenizer),
                "target_vocab_size": self.config.vocab_size,
                "model_max_length": self.config.model_max_length,
                "min_frequency": self.config.min_frequency,
                "special_tokens": {
                    "unk_token": self.config.unk_token,
                    "bos_token": self.config.bos_token,
                    "eos_token": self.config.eos_token,
                    "pad_token": self.config.pad_token,
                    "additional_special_tokens": self.config.special_tokens
                },
                "training_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "config_source": "pre_configurator.py",
                "training_method": "from_scratch"
            }
            
            info_path = self.tokenizer_dir / "training_info.json"
            with info_path.open("w", encoding="utf-8") as f:
                json.dump(training_info, f, ensure_ascii=False, indent=4)
            logger.info("已保存训练信息")
                
        except Exception as e:
            logger.error(f"保存tokenizer失败: {e}")
            raise

    def evaluate(self, test_texts: Optional[List[str]] = None) -> Dict[str, Any]:
        """评估tokenizer性能
        
        Args:
            test_texts: 测试文本列表，如果为None则使用默认测试文本
            
        Returns:
            Dict[str, Any]: 评估结果
        """
        try:
            if not self.tokenizer_dir.exists():
                raise FileNotFoundError(f"Tokenizer目录不存在: {self.tokenizer_dir}")
                
            # 加载训练好的tokenizer
            tokenizer = PreTrainedTokenizerFast.from_pretrained(str(self.tokenizer_dir))
            results = {}
            
            # 测试词表信息
            vocab_size = len(tokenizer)
            results['vocab_info'] = {
                'vocab_size': vocab_size,
                'model_max_length': tokenizer.model_max_length,
                'pad_token': tokenizer.pad_token,
                'eos_token': tokenizer.eos_token,
                'unk_token': tokenizer.unk_token,
                'bos_token': tokenizer.bos_token,
                'tokenizer_type': self.config.tokenizer_type
            }
            logger.info(f"词表大小: {vocab_size}")
            logger.info(f"最大长度: {tokenizer.model_max_length}")
            logger.info(f"Tokenizer类型: {self.config.tokenizer_type}")
            
            # 测试编码解码
            if test_texts is None:
                test_texts = [
                    '我是中国人，我爱我的祖国 @微博 当前数据为测试文件！',
                    'Hello, world! This is a test.',
                    '123456789',
                    '特殊符号：!@#$%^&*()_+-=[]{}|;:,.<>?',
                    'Mixed中英文text with 数字123 and symbols!@#',
                    '<|system|>你好<|user|>测试<|assistant|>回复<|im_start|><|im_end|>'
                ]
            
            results['encoding_tests'] = []
            for i, test_text in enumerate(test_texts):
                try:
                    # 编码
                    encoded = tokenizer(test_text, return_tensors="pt", add_special_tokens=False)
                    tokens = encoded['input_ids'][0].tolist()
                    
                    # 解码
                    decoded = tokenizer.decode(tokens, skip_special_tokens=False)
                    decoded_clean = tokenizer.decode(tokens, skip_special_tokens=True)
                    
                    # 计算字符级别的压缩比
                    char_count = len(test_text)
                    compression_ratio = char_count / len(tokens) if tokens else 0
                    
                    test_result = {
                        'original': test_text,
                        'token_count': len(tokens),
                        'char_count': char_count,
                        'tokens': tokens,
                        'decoded': decoded,
                        'decoded_clean': decoded_clean,
                        'perfect_reconstruction': test_text.strip() == decoded_clean.strip(),
                        'compression_ratio': compression_ratio
                    }
                    results['encoding_tests'].append(test_result)
                    
                    logger.info(f"测试文本 {i+1}: {test_text[:50]}{'...' if len(test_text) > 50 else ''}")
                    logger.info(f"token数量: {len(tokens)}, 字符数: {char_count}")
                    logger.info(f"压缩比: {compression_ratio:.2f}")
                    logger.info(f"完美重构: {test_result['perfect_reconstruction']}")
                    
                except Exception as e:
                    logger.error(f"编码测试失败 (文本 {i+1}): {e}")
                    results['encoding_tests'].append({
                        'original': test_text,
                        'error': str(e)
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"评估过程发生错误: {e}")
            raise

def main():
    """主函数"""
    start_time = time.time()
    
    try:
        # 创建训练器（自动加载配置）
        trainer = BPETokenizerTrainer()
        
        # 训练tokenizer
        success = trainer.train()
        if not success:
            logger.error("Tokenizer训练失败")
            return
        
        # 评估tokenizer
        logger.info("开始评估tokenizer...")
        eval_results = trainer.evaluate()
        logger.info(f"评估结果已保存到: {eval_results}")
        
        total_time = time.time() - start_time
        logger.info(f"程序执行完成，总耗时: {total_time:.2f}秒")
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        raise

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)