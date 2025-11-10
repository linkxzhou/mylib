import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, request, jsonify
from flask_cors import CORS

from llm_factory import LLMFactory, LLMChatAdapter
from embedding_factory import EmbeddingFactory
from get_model_list import get_text_model_list
from util.mylog import logger
from reranker_factory import RerankerFactory
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

def build_chat_prompt(messages: List[Dict[str, Any]]) -> Tuple[str, str, Optional[str]]:
    """
    构造系统信息、对话文本和最后一条消息中的图片URL（如有）
    """
    system_infos: List[str] = []
    conversation_lines: List[str] = []
    last_image: Optional[str] = None

    for msg in messages:
        role = msg.get("role", "")
        text = msg.get("content", "")

        if role == "system":
            if text:
                system_infos.append(text)
            continue

        if role in ("user", "assistant"):
            # 简单拼接上下文（适配器支持 system 单独追加）
            if text:
                conversation_lines.append(f"{role}: {text}")

    system_info = "\n".join(system_infos).strip()
    prompt_text = "\n".join(conversation_lines).strip()
    return system_info, prompt_text, last_image

def create_adapter(
    model: str,
    temperature: Optional[float],
    top_p: Optional[float],
    max_tokens: Optional[int],
) -> LLMChatAdapter:
    # 按照要求：如果 model 包含 `/`，左侧作为 model_type，右侧作为 model；
    # 如果不包含 `/`，直接赋值给 model_type，model 使用默认值 `openai`
    if isinstance(model, str) and "/" in model:
        model_type, model_name = model.split("/", 1)
        model_type = model_type.strip().lower()
        model_name = model_name.strip()
    else:
        model_type = "openai"
        model_name = (model or "").strip().lower()

    logger.info(f"create_adapter: model_type={model_type}, model_name={model_name}")
    llm = LLMFactory.create(
        model_type=model_type,
        model_name=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    return LLMChatAdapter(llm)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "time": int(time.time())})

@app.route("/v1/models", methods=["GET"])
def list_models():
    models = get_text_model_list() or []
    now = int(time.time())
    data = []
    for m in models:
        model_id = m.get("name") or "unknown"
        description = m.get("description", "")
        data.append({
            "id": model_id,
            "object": "model",
            "created": now,
            "owned_by": "proxy",
            "description": description,
        })
    return jsonify({"object": "list", "data": data})

@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    body = request.get_json(silent=True) or {}
    model = body.get("model")
    messages = body.get("messages") or []

    if not model:
        return jsonify({"error": {"message": "model is required"}}), 400

    temperature = body.get("temperature", 0.6)
    top_p = body.get("top_p", 0.9)
    max_tokens = body.get("max_tokens", 32768)

    # 兼容非chat客户端：允许直接传prompt
    if not messages and body.get("prompt"):
        messages = [{"role": "user", "content": body["prompt"]}]

    system_info, prompt_text, image_url = build_chat_prompt(messages)
    adapter = create_adapter(model, temperature, top_p, max_tokens)
    if system_info:
        adapter.appendSystemInfo(system_info)
    isok, result = adapter.chat(prompt_text, image=image_url)
    if not isok:
        return jsonify({"error": {"message": str(result)}}), 500

    return jsonify({
        "id": "chatcmpl-" + uuid.uuid4().hex,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": result},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    })

@app.route("/v1/completions", methods=["POST"])
def completions():
    body = request.get_json(silent=True) or {}
    model = body.get("model")
    prompt = body.get("prompt")
    provider = body.get("provider") or request.headers.get("X-Provider")

    if not model:
        return jsonify({"error": {"message": "model is required"}}), 400
    if prompt is None:
        return jsonify({"error": {"message": "prompt is required"}}), 400

    temperature = body.get("temperature", 0.6)
    top_p = body.get("top_p", 0.9)
    max_tokens = body.get("max_tokens", 32768)

    adapter = create_adapter(model, provider, temperature, top_p, max_tokens)
    # completions 不使用system_info
    isok, result = adapter.chat(str(prompt))
    if not isok:
        return jsonify({"error": {"message": str(result)}}), 500

    return jsonify({
        "id": "cmpl-" + uuid.uuid4().hex,
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "text": result,
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    })

@app.route("/v1/embeddings", methods=["POST"])
def embeddings():
    body = request.get_json(silent=True) or {}
    model = body.get("model")  # 可传入具体HF模型名，否则走默认
    inp = body.get("input")
    if inp is None:
        return jsonify({"error": {"message": "input is required"}}), 400

    # 当前仅huggingface实现；如需接第三方，可在EmbeddingFactory扩展
    kwargs: Dict[str, Any] = body.get("kwargs", {})
    embedder = EmbeddingFactory.create(embedding_type="huggingface", **kwargs)
    vectors: List[List[float]] = []

    if isinstance(inp, list):
        # 多文本
        texts = [str(x) for x in inp]
        vectors = embedder.embed_documents(texts)
    else:
        # 单文本
        vectors = [embedder.embed_query(str(inp))]

    data = [{
        "object": "embedding",
        "index": i,
        "embedding": vec,
    } for i, vec in enumerate(vectors)]

    return jsonify({
        "object": "list",
        "data": data,
        "model": model or "huggingface-default",
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    })

# 新增 rerank 接口
@app.route("/v1/rerank", methods=["POST"])
def rerank():
    body = request.get_json(silent=True) or {}

    query = body.get("query") or body.get("q")
    documents = body.get("documents") or body.get("docs")
    if not query:
        return jsonify({"error": {"message": "query is required"}}), 400
    if not isinstance(documents, list) or len(documents) == 0:
        return jsonify({"error": {"message": "documents must be a non-empty list"}}), 400

    # 解析配置
    reranker_type = (body.get("reranker_type") or "huggingface").strip().lower()
    with_score = body.get("with_score", True)
    top_n = body.get("top_n")
    kwargs: Dict[str, Any] = body.get("kwargs", {})

    # 兼容多种文档对象格式（字符串或带 text/content 字段的对象）
    texts: List[str] = []
    for d in documents:
        if isinstance(d, dict):
            s = d.get("text") or d.get("content") or d.get("document") or d.get("doc")
            texts.append(str(s if s is not None else d))
        else:
            texts.append(str(d))

    try:
        reranker = RerankerFactory.create(reranker_type=reranker_type, **kwargs)
        results = reranker.rerank(query, texts, with_score=with_score)
        if isinstance(top_n, int) and top_n > 0:
            results = results[:top_n]
    except Exception as e:
        return jsonify({"error": {"message": str(e)}}), 500

    data = []
    for i, item in enumerate(results):
        if with_score and isinstance(item, (tuple, list)) and len(item) >= 2:
            doc, score = item[0], item[1]
            data.append({
                "object": "rerank",
                "index": i,
                "document": doc,
                "score": float(score),
            })
        else:
            data.append({
                "object": "rerank",
                "index": i,
                "document": item,
                "score": None,
            })

    return jsonify({
        "object": "list",
        "data": data,
        "reranker_type": reranker_type,
        "query": query,
    })

if __name__ == "__main__":
    port = int(os.getenv("PORT", os.getenv("SCF_HTTP_PORT", "9000")))
    host = os.getenv("HOST", "0.0.0.0")
    app.run(host=host, port=port)