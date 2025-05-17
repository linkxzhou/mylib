import random
import json
import logging
from pathlib import Path
from typing import Iterator, Dict, Any, List
from tokenizers import (
    decoders,
    models,
    pre_tokenizers,
    trainers,
    Tokenizer,
)
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 加载配置
exec(open('0-configurator.py').read())
random.seed(42)

class TokenizerTrainer:
    def __init__(self, base_dir: str, tokenizer_dir: str):
        """初始化TokenizerTrainer"""
        self.base_dir = Path(base_dir)
        self.tokenizer_dir = Path(tokenizer_dir)
        self.special_tokens = ["<unk>", "<s>", "</s>"]
        self.tokenizer = None

    def read_texts_from_jsonl(self, file_path: Path) -> Iterator[str]:
        """读取JSONL文件并提取文本数据"""
        try:
            with file_path.open('r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line)
                        yield data['text']
                    except (json.JSONDecodeError, KeyError) as e:
                        logging.warning(f"第 {line_num} 行解析失败: {e}")
                        continue
        except FileNotFoundError:
            logging.error(f"找不到文件: {file_path}")
            raise

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """验证tokenizer配置"""
        required_keys = ["tokenizer_class", "unk_token", "bos_token", "eos_token"]
        return all(key in config for key in required_keys)

    def create_config(self) -> Dict[str, Any]:
        """创建tokenizer配置"""
        return {
            "add_bos_token": False,
            "add_eos_token": False,
            "add_prefix_space": True,
            "added_tokens_decoder": {
                str(i): {
                    "content": token,
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                } for i, token in enumerate(self.special_tokens)
            },
            "additional_special_tokens": [],
            "bos_token": "<s>",
            "clean_up_tokenization_spaces": False,
            "eos_token": "</s>",
            "legacy": True,
            "model_max_length": 1000000000000000019884624838656,
            "pad_token": None,
            "sp_model_kwargs": {},
            "spaces_between_special_tokens": False,
            "tokenizer_class": "PreTrainedTokenizerFast",
            "unk_token": "<unk>",
            "use_default_system_prompt": False,
            "chat_template": "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}{% if system_message is defined %}{{ system_message }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<s>user\\n' + content + '</s>\\n<s>assistant\\n' }}{% elif message['role'] == 'assistant' %}{{ content + '</s>' + '\\n' }}{% endif %}{% endfor %}"
        }

    def train(self):
        """训练tokenizer并保存相关配置"""
        try:
            # 设置文件路径
            data_path = self.base_dir / 'tokenizer_train.jsonl'
            
            # 初始化tokenizer
            self.tokenizer = Tokenizer(models.BPE())
            self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

            # 设置训练器
            trainer = trainers.BpeTrainer(
                vocab_size=vocab_size,
                special_tokens=special_tokens,
                show_progress=True,
                initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
            )

            # 读取并训练
            logging.info("开始训练tokenizer...")
            texts = self.read_texts_from_jsonl(data_path)
            self.tokenizer.train_from_iterator(texts, trainer=trainer)

            # 设置解码器
            self.tokenizer.decoder = decoders.ByteLevel()

            # 验证特殊token
            for token, idx in zip(self.special_tokens, range(len(self.special_tokens))):
                assert self.tokenizer.token_to_id(token) == idx, f"特殊token {token} 索引错误"

            # 保存tokenizer
            self._save_tokenizer()
            
            logging.info("Tokenizer训练完成并保存")
            
        except Exception as e:
            logging.error(f"训练过程发生错误: {e}")
            raise

    def _save_tokenizer(self):
        """保存tokenizer和配置"""
        try:
            self.tokenizer_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存tokenizer文件
            self.tokenizer.save(str(self.tokenizer_dir / "tokenizer.json"))
            self.tokenizer.model.save(str(self.tokenizer_dir))

            # 创建并验证配置
            config = self.create_config()
            if not self.validate_config(config):
                raise ValueError("配置验证失败")

            # 保存配置
            config_path = self.tokenizer_dir / "tokenizer_config.json"
            with config_path.open("w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
                
        except Exception as e:
            logging.error(f"保存tokenizer失败: {e}")
            raise

    def evaluate(self):
        """评估tokenizer性能"""
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(self.tokenizer_dir))
            
            # 测试chat template
            messages = [
                {"role": "system", "content": "你好，你是中国人，请用中文回答我的问题"},
                {"role": "user", "content": '123'},
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False)
            logging.info(f"Chat template测试:\n{prompt}")
            
            # 测试词表大小
            vocab_size = tokenizer.vocab_size
            actual_size = len(tokenizer)
            logging.info(f"词表大小: {vocab_size}, 实际长度: {actual_size}")
            
            # 测试编码解码
            test_text = '我是中国人，我爱我的祖国 @微博 当前数据为测试文件！'
            tokens = tokenizer(test_text)
            decoded = tokenizer.decode(tokens['input_ids'])
            
            logging.info(f"原文本: {test_text}")
            logging.info(f"token数量: {len(tokens['input_ids'])}")
            logging.info(f"解码结果: {decoded}")
            
        except Exception as e:
            logging.error(f"评估过程发生错误: {e}")
            raise

    def evaluate_gpt2(self):
        """测试GPT2模型"""
        try:
            model = AutoModelForCausalLM.from_pretrained("gpt2")
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            
            prompt = "我是中国人，我爱我的祖国 @微博 当前数据为测试文件！"
            inputs = tokenizer(prompt, return_tensors="pt")
            
            outputs = model.generate(
                inputs.input_ids,
                do_sample=True,
                temperature=0.9,
                max_length=100,
            )
            
            result = tokenizer.batch_decode(outputs)[0]
            logging.info(f"GPT2生成结果: {result}")
            
        except Exception as e:
            logging.error(f"GPT2测试过程发生错误: {e}")
            raise

def main():
    """主函数"""
    try:
        trainer = TokenizerTrainer(base_dir, tokenizer_dir)
        trainer.train()
        trainer.evaluate()
        trainer.evaluate_gpt2()
    except Exception as e:
        logging.error(f"程序执行失败: {e}")
        raise

if __name__ == '__main__':
    main()