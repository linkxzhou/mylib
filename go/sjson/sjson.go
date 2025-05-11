package sjson

import (
	"reflect"
	"strings"
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

// 获取结构体类型的字段信息
func getStructFields(t reflect.Type) []structField {
	if cachedFields, ok := structFieldsCache.Load(t); ok {
		return cachedFields.([]structField)
	}

	numField := t.NumField()
	fields := make([]structField, 0, numField)

	for i := 0; i < numField; i++ {
		f := t.Field(i)
		// 跳过未导出字段
		if f.PkgPath != "" && !f.Anonymous {
			continue
		}

		// 解析json标签
		tag := f.Tag.Get("json")
		if tag == "-" {
			continue
		}

		name := f.Name
		omitempty := false

		if tag != "" {
			// 使用 strings.Cut 替代 strings.Split，减少内存分配
			tagName, options, _ := strings.Cut(tag, ",")
			if tagName != "" {
				name = tagName
			}

			if options != "" {
				// 检查是否包含 omitempty 选项
				for options != "" {
					var opt string
					opt, options, _ = strings.Cut(options, ",")
					if opt == "omitempty" {
						omitempty = true
						break
					}
				}
			}
		}

		fields = append(fields, structField{
			name:      stringToBytes(name),
			index:     i,
			omitempty: omitempty,
			typ:       f.Type,
		})
	}

	structFieldsCache.Store(t, fields)
	return fields
}
