import { Modal } from 'ant-design-vue'
import { h } from 'vue'

// showLoading 修改片段
export function showLoading(title = '加载中', content = '请稍候...') {
    Modal.info({
        title,
        content: toModalContent(content),
        okButtonProps: { style: { display: 'none' } },
        maskClosable: false,
        closable: true,
    })
}

export function hideLoading() {
    // 关闭所有已打开的模态框（包含加载提示）
    Modal.destroyAll()
}

// showError 修改片段
export function showError(title = '错误', content = '') {
    Modal.error({
        title,
        content: toModalContent(content),
        closable: true,
    })
}

// showSuccess 修改片段
export function showSuccess(title = '成功', content = '') {
    Modal.success({
        title,
        content: toModalContent(content),
        closable: true,
    })
}

// 新增：字符串内容到 VNode 的转换，支持 <br> 和 \n 换行
function toModalContent(content) {
    if (content == null) return ''
    if (typeof content !== 'string') return content
    // 如果包含 HTML 标签（如 <br>），按 HTML 渲染
    if (/<[a-z][\s\S]*>/i.test(content)) {
        return h('div', { innerHTML: content })
    }
    // 如果包含 \n，使用 pre 保留换行和空白
    if (content.includes('\n')) {
        return h('pre', { style: 'white-space: pre-wrap; word-break: break-word; margin: 0;' }, content)
    }
    // 普通短文本，直接返回字符串
    return content
}