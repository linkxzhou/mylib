package sjson

import (
	"sync"
)

// 编解码器缓存部分
var (
	// 结构体字段信息缓存
	structFieldsCache sync.Map // map[reflect.Type][]structField
)

// Config 用于配置JSON解析和编码的行为
type Config struct {
	// SortMapKeys 控制对象和map的键是否排序，默认不排序
	SortMapKeys bool
}

// 默认配置
var defaultConfig = Config{
	SortMapKeys: false,
}

// SetDefaultConfig 设置默认的全局配置
func SetDefaultConfig(config Config) {
	defaultConfig = config
}

// GetDefaultConfig 获取当前默认配置
func GetDefaultConfig() Config {
	return defaultConfig
}
