export const SETTINGS_KEY = 'mindlessSettings'

const toNumberOr = (val, fallback) => {
    const n = Number(val)
    return Number.isFinite(n) ? n : fallback
}

export function loadSettings(defaults = {}) {
    try {
        const raw = sessionStorage.getItem(SETTINGS_KEY)
        if (!raw) return { ...defaults }
        const saved = JSON.parse(raw)
        return {
            ...defaults,
            ...saved,
            temperature: toNumberOr(saved?.temperature, toNumberOr(defaults?.temperature, 0.7)),
            depth: toNumberOr(saved?.depth, toNumberOr(defaults?.depth, 3)),
        }
    } catch (e) {
        console.warn('加载设置失败：', e)
        return { ...defaults }
    }
}

export function saveSettings(settings) {
    const payload = {
        ...settings,
        temperature: toNumberOr(settings?.temperature, 0.7),
        depth: toNumberOr(settings?.depth, 3),
    }
    try {
        sessionStorage.setItem(SETTINGS_KEY, JSON.stringify(payload))
    } catch (e) {
        console.error('保存设置失败：', e)
        throw e
    }
}

// 新增：导图数据持久化
export const MINDMAP_KEY = 'mindMapData'

const isValidMindMap = (d) => d && typeof d === 'object' && d.data && typeof d.data === 'object'

export function loadMindMapData(defaults = null) {
    try {
        const raw = sessionStorage.getItem(MINDMAP_KEY)
        if (!raw) return defaults
        const saved = JSON.parse(raw)
        return isValidMindMap(saved) ? saved : (defaults ?? null)
    } catch (e) {
        console.warn('加载导图数据失败：', e)
        return defaults ?? null
    }
}

export function saveMindMapData(mapData) {
    try {
        sessionStorage.setItem(MINDMAP_KEY, JSON.stringify(mapData))
    } catch (e) {
        console.error('保存导图数据失败：', e)
        throw e
    }
}