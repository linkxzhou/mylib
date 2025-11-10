import { thinkingModels } from './const.js'

export function buildPrompt(topic, count, systemPrompt, thinkingModel = 'default') {
    if (!systemPrompt || systemPrompt.trim() === '') systemPrompt = '任意'
    const model = thinkingModels.find(m => m.value === thinkingModel) || thinkingModels[0]
    let examplePrompt = ""
    if (model.example && model.example.trim() !== '') {
        examplePrompt = `
## 思考方式\n请使用 "${model.label}" 思考方式
### 思考样例\n${model.example || ''}`
    }
    return `
## 角色    
现在你是一个 “${topic}” 专家，精通当前领域的百科知识，现在基于知识点 “${topic}” 整理最相关的子知识。

${examplePrompt}

## 输出样例
\`\`\`json
[
  {
    "data": {
      "text": "相关知识点1",
      "note": "相关知识点描述",
      "nextSystemPrompt": "子知识点的提示词"
    },
    "children": []
  },
  {
    "data": {
      "text": "相关知识点2",
      "note": "相关知识点简单描述",
      "nextSystemPrompt": "子知识点的提示词"
    },
    "children": []
  }
  // ...其他
]
\`\`\`
 
## 要求
- 输出JSON
- 知识点思考的要求：${systemPrompt}
- 知识点的数量要求：${count} 个左右，如果重要知识点比较多，可以大于 ${count} 个
- JSON字段\`text\`是知识点，限制在 20 个字以内
- JSON字段\`note\`是知识点的关键词描述，限制在 100-300 个字以内
- JSON字段\`nextSystemPrompt\`是子知识点的提示词，限制在 30-100 个字以内
`
}

export function extractIdeas(raw) {
    // 1) 提取可能存在的内容字段或直接使用字符串
    const content = raw?.choices?.[0]?.message?.content
        ?? raw?.output_text
        ?? raw?.text
        ?? (typeof raw === 'string' ? raw : '')

    // 2) 去掉 ```json 开头与尾部 ```（仅按要求处理首尾围栏）
    const cleaned = String(content)
        .replace(/^\s*```json(?:\s*|\r?\n)/i, '') // 仅替换开头的 ```json（允许后接空白或换行）
        .replace(/```[\s\n]*$/i, '')              // 仅替换结尾的 ```（允许末尾空白/换行）
        .trim()

    // 3) 校验并解析 JSON，不是 JSON 就抛错
    let parsed
    console.log('AI返回:', cleaned)
    try {
        parsed = JSON.parse(cleaned)
    } catch (err) {
        console.error('JSON解析错误:', err)
        throw new Error('返回内容不是有效 JSON')
    }

    return parsed;
}

export function resolveEndpoint(api) {
    return (api || '').trim() + '/chat/completions'
}

export async function requestCompletions({
    api,
    secret,
    model = 'gpt-5-nano',
    temperature = 0.7,
    prompt,
}) {
    const endpoint = resolveEndpoint(api)
    console.log('AI请求', endpoint)

    const headers = { 'Content-Type': 'application/json' }
    if (secret) headers.Authorization = `Bearer ${secret}`

    const body = {
        model,
        messages: [{ role: 'user', content: prompt }],
        temperature
    }

    try {
        const res = await fetch(endpoint, {
            method: 'POST',
            headers,
            body: JSON.stringify(body),
        })
        const isJson = (res.headers.get('content-type') || '').includes('application/json')
        const data = isJson ? await res.json() : await res.text()
        return { data, isJson, endpoint }
    } catch (err) {
        console.error('AI请求失败', err)
        throw err
    }
}