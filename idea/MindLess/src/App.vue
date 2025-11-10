<template>
    <div class="toolbar">
        <!-- toolbar-inner 区块 -->
        <div class="toolbar-inner">
            <!-- 缩放控件：- 100% + -->
            <div class="zoom-control">
                <a-button size="small" shape="circle" :icon="h(MinusOutlined)" @click="zoomOut" />
                <span class="zoom-percent">{{ Math.round(zoom * 100) }}%</span>
                <a-button size="small" shape="circle" :icon="h(PlusOutlined)" @click="zoomIn" />
            </div>

            <a-button :icon="h(PlusOutlined)" :style="{ padding: '4px 10px' }" @click="newMap">新建</a-button>
            <a-button :icon="h(ArrowLeftOutlined)" :style="{ padding: '4px 10px' }" @click="back">返回</a-button>
            <a-button :icon="h(ArrowRightOutlined)" :style="{ padding: '4px 10px' }" @click="forward">前进</a-button>
            <a-button type="primary" danger :icon="h(DeleteOutlined)" :style="{ padding: '4px 10px' }" @click="removeSelected">删除</a-button>
            <a-button :icon="h(SettingOutlined)" :style="{ padding: '4px 10px' }" @click="toggleSettings">设置</a-button>
            <a-button type="primary" :icon="h(BulbOutlined)" :style="{ padding: '4px 10px' }" @click="aiGenerate">AI生成</a-button>
        </div>
    </div>
    <div id="mindMapContainer"></div>
    <div v-if="settingsOpen" class="settings-panel">
        <div class="panel-body">
            <label class="field">
                <span>API：</span>
                <a-input v-model:value="settings.api" placeholder="例如：API地址或密钥" />
            </label>
            <label class="field" style="flex-direction: row; align-items: center; gap: 8px;">
                <span style="white-space: nowrap;">秘钥：</span>
                <a-input 
                    v-model:value="settings.secret" 
                    placeholder="例如：sk-..."
                    style="flex: 1; min-width: 0;"    
                />
            </label>

            <!-- 模型：改为一行显示 -->
            <label class="field" style="flex-direction: row; align-items: center; gap: 8px;">
                <span style="white-space: nowrap;">模型：</span>
                <a-input
                    v-model:value="settings.model"
                    placeholder="例如：gpt-4o-mini"
                    style="flex: 1; min-width: 0;"
                />
            </label>

            <!-- 新增：模式开关（专注/自由） -->
            <label class="field" style="flex-direction: row; align-items: center; gap: 8px;">
                <span style="white-space: nowrap;">模式：</span>
                <a-switch
                    v-model:checked="settings.focusMode"
                    checked-children="专注"
                    un-checked-children="自由"
                    :style="{ width: '60px' }"
                />
            </label>

            <label class="field" style="flex-direction: row; align-items: center; gap: 8px;">
                <span style="white-space: nowrap;">生成子节点个数（范围值）：</span>
                <a-input-number
                    v-model:value="settings.depth"
                    :min="1"
                    :max="10"
                    :step="1"
                    style="flex: 0 0 auto; width: 120px;"
                />
            </label>

            <!-- 新增：思考模型选择 -->
            <label class="field" style="flex-direction: row; align-items: center; gap: 8px;">
                <span style="white-space: nowrap;">思考方式：</span>
                <a-select
                    v-model:value="settings.thinkingModel"
                    :options="thinkingModels"
                    style="flex: 0 0 auto; min-width: 200px;"
                    placeholder="请选择思考模型"
                />
            </label>

            <label class="field">
                <span>系统提示词：</span>
                <a-textarea
                    v-model:value="settings.systemPrompt"
                    placeholder="输入系统提示词"
                    :auto-size="{ minRows: 2, maxRows: 5 }"
                />
            </label>

            <label class="field">
                <span>布局：</span>
            </label>
            <div class="chart-list">
                <a-button
                    v-for="l in layouts"
                    :key="l.key"
                    size="small"
                    @click="applyLayout(l.key)"
                >
                    {{ l.name }}
                </a-button>
            </div>

            <div class="field">
                <span>说明：<br>
                    <br>- 专注模式：每次基于当前节点的专属提示词进行后续生成。
                    <br>- 自由模式：始终使用全局系统提示词，可以按照自己的想法随时修改。
                    <br>- 思考方式：{{ currentThinkingModel?.label || settings.thinkingModel }}。
                    <br><span style="white-space: pre-wrap;">{{ currentThinkingModel?.example || '' }}</span>
                </span>
            </div>
        </div>
        <div class="panel-actions">
            <a-button type="primary" @click="saveSettings">保存</a-button>
            <a-button style="margin-left: 8px" type="primary" danger @click="settingsOpen = false">关闭</a-button>
        </div>
    </div>
</template>

<style>
/** @import "./simpleMindMap.esm.css"; **/
</style>

<script setup>
// script setup 导入区块
// 引入 Ant Design Vue 组件
import { Button as AButton, Input as AInput, InputNumber as AInputNumber, Textarea as ATextarea, Switch as ASwitch, Select as ASelect } from 'ant-design-vue'
// 引入图标（新增 MinusOutlined）
import { MinusOutlined, PlusOutlined, ArrowLeftOutlined, ArrowRightOutlined, DeleteOutlined, SettingOutlined, BulbOutlined } from '@ant-design/icons-vue'
import { ref, onMounted, watch, h, computed } from 'vue'
import MindMap from "simple-mind-map"
import { showLoading, hideLoading, showError } from './modal.js'
import { buildPrompt as libBuildPrompt, extractIdeas as libExtractIdeas, requestCompletions } from './libai.js'
import { loadSettings as loadSettingsFromStorage, saveSettings as saveSettingsToStorage, loadMindMapData, saveMindMapData } from './storage.js'
import { thinkingModels as thinkingModelOptions, layouts as layoutOptions } from './const.js'
const mindMapRef = ref(null)
const activeNodes = ref([])
const settingsOpen = ref(false)
const settings = ref({
    api: '',
    secret: '',
    model: 'gpt-5-nano',
    temperature: 0.7,
    systemPrompt: '',
    depth: 3,
    focusMode: true,
    thinkingModel: 'first-principles' // 新增：默认选择第一性原理
})

// 思考模型选项列表
const thinkingModels = thinkingModelOptions

const currentThinkingModel = computed(() => {
    const v = settings.value.thinkingModel
    return thinkingModels.find(m => m.value === v) || { label: v, value: v, example: '' }
})

// 使用浏览器 sessionStorage 读取/保存设置
const loadSettings = () => {
    try {
        settings.value = loadSettingsFromStorage(settings.value)
    } catch (e) {
        console.warn('加载设置失败：', e)
    }
}

const saveSettings = () => {
    try {
        saveSettingsToStorage(settings.value)
        console.log('设置已保存到 sessionStorage')
    } catch (e) {
        console.error('保存设置失败：', e)
    }
    // 保存后隐藏设置面板
    settingsOpen.value = false
}

const getNodeText = (node) => node?.data?.text || (node?.getData?.()?.text) || ''
const getNodeSystemPrompt = (node) => node?.data?.nextSystemPrompt || (node?.getData?.()?.nextSystemPrompt) || ''

const newMap = () => {
    if (!mindMapRef.value) return
    mindMapRef.value.setData({ data: { text: '根节点' }, children: [] })
    mindMapRef.value.view.reset()
}

const back = () => {
    if (!mindMapRef.value) return
    mindMapRef.value.execCommand('BACK')
}

const forward = () => {
    if (!mindMapRef.value) return
    mindMapRef.value.execCommand('FORWARD')
}

const removeSelected = () => {
    if (!mindMapRef.value) return
    mindMapRef.value.execCommand('REMOVE_NODE')
}

const aiGenerate = async () => {
    if (!mindMapRef.value) return
    const baseNode = activeNodes.value?.[0]
    const baseText = getNodeText(baseNode)
    if (!baseText || baseText.trim().length === 0) {
        showError('请先选择一个节点或者输入一个主题')
        return
    }

    // 新增：按模式选择系统提示词
    let nodeSystemPrompt
    if (settings.value.focusMode) {
        nodeSystemPrompt = getNodeSystemPrompt(baseNode)
        if (!nodeSystemPrompt || nodeSystemPrompt.trim().length === 0) {
            nodeSystemPrompt = settings.value.systemPrompt
        }
    } else {
        nodeSystemPrompt = settings.value.systemPrompt
    }

    const count = Math.max(1, Math.min(10, Number(settings.value.depth) || 3))
    const prompt = libBuildPrompt(
        baseText,
        count,
        nodeSystemPrompt,
        settings.value.thinkingModel || 'default',
    )

    showLoading('AI生成中...', `当前模式：${settings.value.focusMode ? '专注模式' : '普通模式'}
知识点方向：${nodeSystemPrompt}

Prompt:
    ${prompt}`)
    try {
        const { data } = await requestCompletions({
            api: settings.value.api,
            secret: settings.value.secret,
            model: settings.value.model || 'gpt-5-nano',
            temperature: settings.value.temperature,
            prompt,
        })

        const ideas = libExtractIdeas(data, count)
        console.log('解析到子节点：', JSON.stringify(ideas))
        if (ideas.length) {
            mindMapRef.value.execCommand('INSERT_MULTI_CHILD_NODE', [], ideas)
            hideLoading()
        } else {
            hideLoading()
            showError('AI返回内容为空或未解析到子节点')
        }
    } catch (err) {
        hideLoading()
        const msg = err?.message || String(err)
        showError(`AI生成失败：${msg}`)
        console.error('AI生成失败：', err)
    }
}

const toggleSettings = () => {
    settingsOpen.value = !settingsOpen.value
}

const layouts = layoutOptions

const applyLayout = (key) => {
    if (!mindMapRef.value) return
    mindMapRef.value.setLayout(key)
    mindMapRef.value.view.reset()
}

const zoom = ref(1)

const applyZoom = (next) => {
    const mm = mindMapRef.value
    const clamped = Math.min(2, Math.max(0.2, Number(next) || 1))
    zoom.value = clamped
    if (!mm) return
    const v = mm.view

    // 优先调用库方法（若存在）
    if (v && typeof v.setScale === 'function') {
        v.setScale(clamped)
        return
    }

    if (v && typeof v.scale === 'function') {
        // 有些库用 scale(value) 设定缩放
        v.scale(clamped)
        return
    }

    const el = document.getElementById('mindMapContainer')
    if (el) {
        el.style.transform = `scale(${clamped})`
        el.style.transformOrigin = 'top left'
    }
}

const zoomIn = () => applyZoom(zoom.value + 0.1)
const zoomOut = () => applyZoom(zoom.value - 0.1)

onMounted(() => {
    loadSettings()
    // 使用已保存的导图数据作为初始数据；若无则回落到示例数据
    const initialData = loadMindMapData({
        data: { text: '主题' },
        children: []
    })

    const mindMap = new MindMap({
        el: document.getElementById('mindMapContainer'),
        enableFreeDrag: true,
        mousewheelAction: 'zoom',
        mousewheelZoomActionReverse: true,
        data: initialData
    });
    mindMapRef.value = mindMap

    // 初始化缩放
    try {
        const v = mindMapRef.value?.view
        const initialScale = (v && typeof v.scale === 'number') ? v.scale : 1
        applyZoom(initialScale || 1)
    } catch {
        applyZoom(1)
    }

    mindMap.on('node_active', (node, activeNodeList) => {
        activeNodes.value = activeNodeList || (node ? [node] : [])
    })

    // 数据变更时持久化到 sessionStorage
    mindMap.on('data_change', (data) => {
        try {
            saveMindMapData(data)
        } catch (e) {
            console.warn('写入 sessionStorage 失败：', e)
        }
    })
})
</script>