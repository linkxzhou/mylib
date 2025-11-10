# LLM API 框架

一个灵活的统一接口框架，用于与各种大语言模型API进行交互。

## 主要特点

- 为多种LLM提供商提供统一接口
- 支持多种LLM提供商，包括：
  - OpenAI
  - 百度千帆
  - 阿里巴巴通义千问
  - 智谱AI
  - Ollama
  - SiliconFlow
  - HuggingFace
- 插件系统用于输入和输出的预处理/后处理
- 健壮的错误处理和日志记录
- 跨提供商的自动模型发现

## 使用方法

### 基本用法

```python
from llm.llmapi.llm_factory import LLMFactory, LLMChatAdapter

# 创建LLM实例
llm = LLMFactory.create(
    model_type="openai",  # 可选: "openai", "qianfan", "qwen", "zhipu", "ollama", "siliconflow"
    model_name="gpt-3.5-turbo",  # 特定提供商的模型名称
    temperature=0.7,  # 控制随机性 (0.0-1.0)
    top_p=0.9,        # 核采样参数 (0.0-1.0)
    max_tokens=1000   # 生成的最大token数
)

# 创建适配器以便更轻松地交互
adapter = LLMChatAdapter(llm)

# 可选: 添加系统指令
adapter.appendSystemInfo("你是一个有帮助的助手。")

# 与模型对话
success, response = adapter.chat("告诉我关于人工智能的信息。")
if success:
    print(response)
else:
    print(f"错误: {response}")
```

### 获取可用模型列表

```python
from llm.llmapi.get_model_list import get_text_model_list

# 获取所有提供商的可用文本模型列表
models = get_text_model_list()
for model in models:
    print(f"名称: {model['name']}, 描述: {model.get('description', 'N/A')}")
```

## 插件系统

该框架包含用于处理输入和输出的插件系统：

```python
from llm.llmapi.llm_plugin import Plugin

# 创建自定义插件
class MyCustomPlugin(Plugin):
    def process_input(self, prompt: str) -> str:
        # 预处理用户输入
        return prompt.strip()
    
    def process_output(self, response: str) -> str:
        # 后处理模型响应
        return response.replace("AI", "人工智能")

# 添加到适配器
from llm.llmapi.llm_factory import LLMChatAdapter
adapter = LLMChatAdapter(llm)
adapter.plugins.append(MyCustomPlugin())
```

## 内置插件

- `ReplaceHtmlPlugin`: 转义模型响应中的HTML标签
- `ReplaceImagePlugin`: 处理响应中的图片路径，将本地图片转换为base64格式

## 提供商配置

每个提供商需要特定的配置（API密钥、端点等），应按照其各自的文档进行设置，环境配置：
```
export PYTHONPATH=`pwd`:$PYTHONPATH
export QIANFAN_API_KEY=
export ZHIPU_API_KEY=
export OPENAI_API_KEY=
export OPENAI_API_BASE=
export DASHSCOPE_API_KEY=
export DASHSCOPE_API_BASE=
# 可选设置
export HUGGINGFACE_CACHE_DIR=
export HUGGINGFACE_MODEL_PATH=
export OLLAMA_API_BASE=
export SILICONFLOW_API_KEY=
```

## OpenAI API 代理接口

该项目提供一个基于 FastAPI 的 OpenAI 兼容代理接口，统一接入多家模型提供商（OpenAI、千帆、通义、智谱、Ollama、SiliconFlow、HuggingFace），并复用本仓库的 LLMFactory/EmbeddingFactory。

- 基础端点：
  - `GET /health`
  - `GET /v1/models`（跨提供商模型发现）
  - `POST /v1/chat/completions`（OpenAI Chat API）
  - `POST /v1/completions`（Legacy Text Completions）
  - `POST /v1/embeddings`（OpenAI Embeddings API；当前接 HuggingFace）

- 运行
  1) 安装依赖：
  ```bash
  pip install flask flask-cors
  ```

- 测试用例：
```
curl -X POST http://localhost:18089/v1/embeddings -H "Content-Type: application/json" -d '{"model":"BAAI/bge-small-zh-v1.5","input":["你好世界","人工智能很有趣"]}'

curl http://localhost:18089/v1/models

curl -X POST http://localhost:18089/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"gpt-5","messages":[{"role":"system","content":"你是一个有帮助的助手"},{"role":"user","content":[{"type":"text","text":"请做个自我介绍"}]}],"temperature":0.6}'

curl -X POST http://localhost:18089/v1/completions -H "Content-Type: application/json" -d '{"model":"gpt-5","prompt":"用中文写一首关于海洋的俳句。","temperature":0.7}'

curl -X POST http://localhost:18089/v1/rerank -H "Content-Type: application/json" -d '{"query":"Python 学习路线","documents":[{"text":"从基础语法开始学习 Python"},{"content":"掌握数据结构与算法"}],"with_score":true}'
```
