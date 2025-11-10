# 数据来源
tokenizer训练数据：https://huggingface.co/datasets/Mxode/Chinese-Instruct/tree/main/chinese-medical

# 安装库

## 安装conda
yum install -y conda

## 启动conda
conda create --name myenv python=3.11

## 切换环境
conda activate myenv

## 安装库
```
pip install psutil
pip install ujson
pip install jsonlines
pip install transformers
pip install torch
pip install trl
pip install peft
```

## 训练
（1）预训练：`accelerate launch --num_processes=4 3-pretrain.py`

## 数据集

### 1. 预训练数据

### 2. SFT数据
（1）https://huggingface.co/collections/liang-sh/sft-data-6555b26d11b77e167f86eb0c

### 参考
（1）https://zhuanlan.zhihu.com/p/715570503
（2）https://github.com/lonePatient/awesome-pretrained-chinese-nlp-models?tab=readme-ov-file
（3）https://www.modelscope.cn/datasets/gongjy/minimind_dataset
（4）https://github.com/brightmart/nlp_chinese_corpus?tab=readme-ov-file

### 小模型
（1）https://huggingface.co/delphi-suite

## 参考
（1）https://github.com/AI-Study-Han/Zero-Qwen-VL
（2）https://github.com/jiahe7ay/MINI_LLM
（3）https://github.com/wei-potato/Train-llm-from-scratch
（4）https://github.com/DLLXW/baby-llama2-chinese
（5）https://zhuanlan.zhihu.com/p/624412809
（6）https://github.com/liguodongiot/llm-action
（7）https://github.com/charent/ChatLM-mini-Chinese?tab=readme-ov-file