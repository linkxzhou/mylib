# 翻译代理使用说明

## 安装

首先，确保安装必要的库：

```bash
pip install langchain gradio
# 安装所需的LLM API库，例如：
# pip install qianfan-sdk
# 或者安装所有依赖：
pip install -r requirements.txt
```

## 基本用法

```python
from llm.agent.translation import TranslationAgent
from llmapi.llm_factory import LLMFactory, LLMChatAdapter

# 初始化大语言模型
llm = LLMFactory.create("qianfan", model_name="deepseek-v3")  # 可替换为其他模型

# 初始化翻译代理
translator = TranslationAgent(llm)

# 创建LLM适配器
llm_chat_adapter = LLMChatAdapter(llm)

# 执行翻译
result = translator.translate(
    source_lang="English",  # 源语言
    target_lang="Chinese",  # 目标语言
    source_text="Your text to translate goes here.",  # 待翻译文本
    llm_chat=llm_chat_adapter  # LLM聊天适配器
)

print(f"原文: {source_text}")
print(f"翻译结果: {result}")
```

## 地区语言适配

```python
# 指定特定国家/地区的语言风格
result = translator.translate(
    source_lang="English",
    target_lang="Chinese",
    source_text="Your text to translate goes here.",
    country="Taiwan",  # 指定地区，如"Taiwan"、"Singapore"等
    llm_chat=llm_chat_adapter
)
```

## 工作原理

翻译过程分为三个主要步骤：

1. **初始翻译**：利用LLM进行第一轮基础翻译
2. **翻译反思**：系统分析初次翻译结果，从准确性、流畅度、风格和术语四个维度提出改进建议
3. **翻译改进**：根据反思结果对初次翻译进行优化，生成最终翻译结果

## 评估维度

- **准确性**：修正添加、误译、遗漏或未翻译的内容
- **流畅度**：应用目标语言的语法、拼写和标点规则，避免不必要的重复
- **风格**：确保翻译反映原文风格并考虑文化背景
- **术语**：确保术语使用一致且反映源文本领域，使用目标语言中的等效习语

## 支持的模型

通过LLMFactory可以使用多种大模型API：
- 千帆大模型（如文档中示例的deepseek-v3）
- 百度文心
- OpenAI模型
- 其他支持的LLM API

## Gradio Web界面

### 快速启动

```bash
# 启动简单翻译界面
python run_ui.py

# 启动完整Agent平台
python run_ui.py --mode full

# 自定义端口
python run_ui.py --port 8080

# 启用公网分享
python run_ui.py --share
```

### 界面功能

- **智能翻译**：支持多种语言互译
- **模型选择**：支持千帆、通义千问、智谱、OpenAI等多种模型
- **地区适配**：支持不同地区的语言风格
- **实时预览**：即时显示翻译结果
- **示例模板**：提供常用翻译示例

### 界面截图

访问 `http://localhost:7860` 即可使用Web界面进行翻译。

## 扩展Agent

本框架支持扩展其他类型的Agent：

```python
from agent_framework import BaseAgent, AgentConfig

class CustomAgent(BaseAgent):
    def __init__(self):
        config = AgentConfig(
            name="自定义助手",
            description="自定义功能描述",
            icon="🔧"
        )
        super().__init__(config)
    
    def initialize(self, **kwargs):
        # 初始化逻辑
        return True, "初始化成功"
    
    def create_interface(self):
        # 创建Gradio界面
        pass
    
    def process(self, *args, **kwargs):
        # 处理用户输入
        pass
```

## 系统要求

- Python 3.7+
- langchain
- gradio
- 相应的LLM API和SDK