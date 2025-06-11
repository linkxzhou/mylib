import random
import json
import logging
from pathlib import Path
from typing import Iterator, Dict, Any, List, Optional, Tuple
from tokenizers import (
    decoders,
    models,
    pre_tokenizers,
    trainers,
    Tokenizer,
)
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import time
from dataclasses import dataclass

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('tokenizer_training.log', encoding='utf-8')
    ]
)

# 加载配置
try:
    exec(open('stage0-configurator.py').read())
except FileNotFoundError:
    logging.warning("配置文件未找到，使用默认配置")
    # 默认配置
    vocab_size = 19200
    special_tokens = ["<unk>", "<s>", "</s>"]
    base_dir = "../datasets"
    tokenizer_dir = "./my_tokenizer"

random.seed(42)

@dataclass
class TokenizerConfig:
    """Tokenizer配置类"""
    vocab_size: int = 19200
    special_tokens: List[str] = None
    add_prefix_space: bool = True
    show_progress: bool = True
    model_max_length: int = 2048  # 更合理的最大长度
    
    def __post_init__(self):
        if self.special_tokens is None:
            self.special_tokens = ["<unk>", "<s>", "</s>"]

class TokenizerTrainer:
    def __init__(self, base_dir: str, tokenizer_dir: str, config: Optional[TokenizerConfig] = None):
        """初始化TokenizerTrainer
        
        Args:
            base_dir: 数据目录路径
            tokenizer_dir: tokenizer保存目录
            config: tokenizer配置，如果为None则使用默认配置
        """
        self.base_dir = Path(base_dir)
        self.tokenizer_dir = Path(tokenizer_dir)
        self.config = config or TokenizerConfig()
        self.tokenizer: Optional[Tokenizer] = None
        
        # 验证目录
        if not self.base_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {self.base_dir}")
            
        logging.info(f"初始化TokenizerTrainer: base_dir={self.base_dir}, tokenizer_dir={self.tokenizer_dir}")
        logging.info(f"配置: vocab_size={self.config.vocab_size}, special_tokens={self.config.special_tokens}")

    def read_texts_from_jsonl(self, file_path: Path, max_lines: Optional[int] = None) -> Iterator[str]:
        """读取JSONL文件并提取文本数据
        
        Args:
            file_path: JSONL文件路径
            max_lines: 最大读取行数，None表示读取全部
            
        Yields:
            str: 文本内容
        """
        if not file_path.exists():
            raise FileNotFoundError(f"找不到文件: {file_path}")
            
        total_lines = 0
        valid_lines = 0
        
        try:
            with file_path.open('r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if max_lines and line_num > max_lines:
                        break
                        
                    total_lines += 1
                    line = line.strip()
                    if not line:
                        continue
                        
                    try:
                        data = json.loads(line)
                        if 'text' in data and data['text'].strip():
                            valid_lines += 1
                            yield data['text']
                    except (json.JSONDecodeError, KeyError) as e:
                        logging.warning(f"第 {line_num} 行解析失败: {e}")
                        continue
                        
            logging.info(f"读取完成: 总行数={total_lines}, 有效行数={valid_lines}")
            
        except Exception as e:
            logging.error(f"读取文件失败: {e}")
            raise

    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """验证tokenizer配置
        
        Args:
            config: 配置字典
            
        Returns:
            Tuple[bool, List[str]]: (是否有效, 错误信息列表)
        """
        required_keys = ["tokenizer_class", "unk_token", "bos_token", "eos_token"]
        missing_keys = [key for key in required_keys if key not in config]
        
        errors = []
        if missing_keys:
            errors.append(f"缺少必需的配置项: {missing_keys}")
            
        # 验证特殊token是否在配置中
        for token in self.config.special_tokens:
            if token not in [config.get("unk_token"), config.get("bos_token"), config.get("eos_token")]:
                if token not in config.get("additional_special_tokens", []):
                    errors.append(f"特殊token {token} 未在配置中定义")
                    
        return len(errors) == 0, errors

    def create_config(self) -> Dict[str, Any]:
        """创建tokenizer配置"""
        # 动态构建added_tokens_decoder
        added_tokens_decoder = {}
        for i, token in enumerate(self.config.special_tokens):
            added_tokens_decoder[str(i)] = {
                "content": token,
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            }
            
        return {
            "add_bos_token": False,
            "add_eos_token": False,
            "add_prefix_space": self.config.add_prefix_space,
            "added_tokens_decoder": added_tokens_decoder,
            "additional_special_tokens": [],
            "bos_token": "<s>",
            "clean_up_tokenization_spaces": False,
            "eos_token": "</s>",
            "legacy": True,
            "model_max_length": self.config.model_max_length,
            "pad_token": None,
            "sp_model_kwargs": {},
            "spaces_between_special_tokens": False,
            "tokenizer_class": "PreTrainedTokenizerFast",
            "unk_token": "<unk>",
            "use_default_system_prompt": False,
            "chat_template": "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}{% if system_message is defined %}{{ system_message }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<s>user\\n' + content + '</s>\\n<s>assistant\\n' }}{% elif message['role'] == 'assistant' %}{{ content + '</s>' + '\\n' }}{% endif %}{% endfor %}"
        }

    def train(self, data_file: str = 'tokenizer_train.jsonl', max_lines: Optional[int] = None) -> bool:
        """训练tokenizer并保存相关配置
        
        Args:
            data_file: 训练数据文件名
            max_lines: 最大读取行数，用于测试或小规模训练
            
        Returns:
            bool: 训练是否成功
        """
        start_time = time.time()
        
        try:
            # 设置文件路径
            data_path = self.base_dir / data_file
            if not data_path.exists():
                raise FileNotFoundError(f"训练数据文件不存在: {data_path}")
            
            logging.info(f"开始训练tokenizer，数据文件: {data_path}")
            
            # 初始化tokenizer
            self.tokenizer = Tokenizer(models.BPE())
            self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
                add_prefix_space=self.config.add_prefix_space
            )

            # 设置训练器
            trainer = trainers.BpeTrainer(
                vocab_size=self.config.vocab_size,
                special_tokens=self.config.special_tokens,
                show_progress=self.config.show_progress,
                initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
            )

            # 读取并训练
            logging.info("开始读取训练数据...")
            texts = self.read_texts_from_jsonl(data_path, max_lines)
            
            logging.info("开始训练tokenizer...")
            train_start = time.time()
            self.tokenizer.train_from_iterator(texts, trainer=trainer)
            train_time = time.time() - train_start
            logging.info(f"训练完成，耗时: {train_time:.2f}秒")

            # 设置解码器
            self.tokenizer.decoder = decoders.ByteLevel()

            # 验证特殊token
            self._validate_special_tokens()

            # 保存tokenizer
            self._save_tokenizer()
            
            total_time = time.time() - start_time
            logging.info(f"Tokenizer训练完成并保存，总耗时: {total_time:.2f}秒")
            return True
            
        except Exception as e:
            logging.error(f"训练过程发生错误: {e}")
            return False
            
    def _validate_special_tokens(self) -> None:
        """验证特殊token的索引"""
        logging.info("验证特殊token索引...")
        for token, expected_idx in zip(self.config.special_tokens, range(len(self.config.special_tokens))):
            actual_idx = self.tokenizer.token_to_id(token)
            if actual_idx != expected_idx:
                raise ValueError(f"特殊token {token} 索引错误: 期望={expected_idx}, 实际={actual_idx}")
        logging.info("特殊token验证通过")

    def _save_tokenizer(self) -> None:
        """保存tokenizer和配置"""
        try:
            # 创建目录
            self.tokenizer_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"保存tokenizer到: {self.tokenizer_dir}")
            
            # 保存tokenizer文件
            tokenizer_json_path = self.tokenizer_dir / "tokenizer.json"
            self.tokenizer.save(str(tokenizer_json_path))
            logging.info(f"已保存tokenizer.json")
            
            # 保存模型文件
            self.tokenizer.model.save(str(self.tokenizer_dir))
            logging.info(f"已保存模型文件")

            # 创建并验证配置
            config = self.create_config()
            is_valid, errors = self.validate_config(config)
            if not is_valid:
                error_msg = f"配置验证失败: {'; '.join(errors)}"
                logging.error(error_msg)
                raise ValueError(error_msg)

            # 保存配置
            config_path = self.tokenizer_dir / "tokenizer_config.json"
            with config_path.open("w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
            logging.info(f"已保存tokenizer_config.json")
            
            # 保存训练信息
            info = {
                "vocab_size": self.config.vocab_size,
                "special_tokens": self.config.special_tokens,
                "training_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model_max_length": self.config.model_max_length
            }
            info_path = self.tokenizer_dir / "training_info.json"
            with info_path.open("w", encoding="utf-8") as f:
                json.dump(info, f, ensure_ascii=False, indent=4)
            logging.info(f"已保存训练信息")
                
        except Exception as e:
            logging.error(f"保存tokenizer失败: {e}")
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
                
            tokenizer = AutoTokenizer.from_pretrained(str(self.tokenizer_dir))
            results = {}
            
            # 测试chat template
            try:
                messages = [
                    {"role": "system", "content": "你好，你是中国人，请用中文回答我的问题"},
                    {"role": "user", "content": '123'},
                ]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False)
                results['chat_template_test'] = prompt
                logging.info(f"Chat template测试:\n{prompt}")
            except Exception as e:
                logging.warning(f"Chat template测试失败: {e}")
                results['chat_template_test'] = f"失败: {e}"
            
            # 测试词表大小
            vocab_size = tokenizer.vocab_size
            actual_size = len(tokenizer)
            results['vocab_info'] = {
                'vocab_size': vocab_size,
                'actual_size': actual_size,
                'size_match': vocab_size == actual_size
            }
            logging.info(f"词表大小: {vocab_size}, 实际长度: {actual_size}")
            
            # 测试编码解码
            if test_texts is None:
                test_texts = [
                    '我是中国人，我爱我的祖国 @微博 当前数据为测试文件！',
                    'Hello, world! This is a test.',
                    '123456789',
                    '特殊符号：!@#$%^&*()_+-=[]{}|;:,.<>?'
                ]
            
            results['encoding_tests'] = []
            for i, test_text in enumerate(test_texts):
                try:
                    tokens = tokenizer(test_text)
                    decoded = tokenizer.decode(tokens['input_ids'])
                    
                    test_result = {
                        'original': test_text,
                        'token_count': len(tokens['input_ids']),
                        'decoded': decoded,
                        'perfect_reconstruction': test_text == decoded,
                        'compression_ratio': len(test_text) / len(tokens['input_ids']) if tokens['input_ids'] else 0
                    }
                    results['encoding_tests'].append(test_result)
                    
                    logging.info(f"测试文本 {i+1}: {test_text}")
                    logging.info(f"token数量: {len(tokens['input_ids'])}")
                    logging.info(f"解码结果: {decoded}")
                    logging.info(f"完美重构: {test_result['perfect_reconstruction']}")
                    logging.info(f"压缩比: {test_result['compression_ratio']:.2f}")
                    
                except Exception as e:
                    logging.error(f"编码测试失败 (文本 {i+1}): {e}")
                    results['encoding_tests'].append({
                        'original': test_text,
                        'error': str(e)
                    })
            
            return results
            
        except Exception as e:
            logging.error(f"评估过程发生错误: {e}")
            raise

    def evaluate_gpt2(self, prompt: str = "我是中国人，我爱我的祖国 @微博 当前数据为测试文件！") -> Optional[str]:
        """测试GPT2模型
        
        Args:
            prompt: 测试提示文本
            
        Returns:
            Optional[str]: 生成结果，失败时返回None
        """
        try:
            logging.info("开始GPT2模型测试...")
            model = AutoModelForCausalLM.from_pretrained("gpt2")
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            
            # 设置pad_token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            inputs = tokenizer(prompt, return_tensors="pt")
            
            outputs = model.generate(
                inputs.input_ids,
                do_sample=True,
                temperature=0.9,
                max_length=100,
                pad_token_id=tokenizer.pad_token_id
            )
            
            result = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            logging.info(f"GPT2生成结果: {result}")
            return result
            
        except Exception as e:
            logging.error(f"GPT2测试过程发生错误: {e}")
            return None

def main():
    """主函数"""
    start_time = time.time()
    
    try:
        # 创建配置
        config = TokenizerConfig(
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            add_prefix_space=True,
            show_progress=True
        )
        
        # 创建训练器
        trainer = TokenizerTrainer(base_dir, tokenizer_dir, config)
        
        # 训练tokenizer
        success = trainer.train()
        if not success:
            logging.error("Tokenizer训练失败")
            return False
        
        # 评估tokenizer
        logging.info("开始评估tokenizer...")
        eval_results = trainer.evaluate()
        
        # 保存评估结果
        eval_path = Path(tokenizer_dir) / "evaluation_results.json"
        with eval_path.open("w", encoding="utf-8") as f:
            json.dump(eval_results, f, ensure_ascii=False, indent=4)
        logging.info(f"评估结果已保存到: {eval_path}")
        
        # GPT2测试（可选）
        logging.info("开始GPT2测试...")
        gpt2_result = trainer.evaluate_gpt2()
        if gpt2_result:
            logging.info("GPT2测试完成")
        else:
            logging.warning("GPT2测试失败")
        
        total_time = time.time() - start_time
        logging.info(f"程序执行完成，总耗时: {total_time:.2f}秒")
        return True
        
    except Exception as e:
        logging.error(f"程序执行失败: {e}")
        return False

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)