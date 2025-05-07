package sjson

import (
	"reflect"
	"sort"
	"strings"
	"unicode/utf8"
)

// 高效处理 map 键的函数，尽量避免转义
func writeMapKey(sb *strings.Builder, key string) {
	sb.WriteByte('"')

	// 快速路径：检查是否需要转义
	needsEscape := false
	for i := 0; i < len(key); i++ {
		if c := key[i]; c < utf8.RuneSelf && !safeSet[c] {
			needsEscape = true
			break
		}
	}

	if !needsEscape {
		// 不需要转义，直接写入
		sb.WriteString(key)
		sb.WriteString(`":`)
		return
	}

	// 需要转义的情况
	for i := 0; i < len(key); i++ {
		c := key[i]
		if c < utf8.RuneSelf && !safeSet[c] {
			switch c {
			case '"':
				sb.WriteString(`\"`)
			case '\\':
				sb.WriteString(`\\`)
			case '\b':
				sb.WriteString(`\b`)
			case '\f':
				sb.WriteString(`\f`)
			case '\n':
				sb.WriteString(`\n`)
			case '\r':
				sb.WriteString(`\r`)
			case '\t':
				sb.WriteString(`\t`)
			default:
				// 小于32的控制字符需要转义为\uXXXX
				if c < 32 {
					sb.WriteString(unicodeHex[c])
				} else {
					sb.WriteByte(c)
				}
			}
		} else {
			sb.WriteByte(c)
		}
	}
	sb.WriteString(`":`)
}

// 添加高效处理 map 键的函数（基于字节切片），尽量避免转义
func appendMapKey(buf []byte, key string) []byte {
	buf = append(buf, '"')

	// 快速路径：检查是否需要转义
	needsEscape := false
	for i := 0; i < len(key); i++ {
		if c := key[i]; c < utf8.RuneSelf && !safeSet[c] {
			needsEscape = true
			break
		}
	}

	if !needsEscape {
		// 不需要转义，直接写入
		buf = append(buf, key...)
		buf = append(buf, '"', ':')
		return buf
	}

	// 需要转义的情况
	for i := 0; i < len(key); i++ {
		c := key[i]
		if c < utf8.RuneSelf && !safeSet[c] {
			switch c {
			case '"':
				buf = append(buf, '\\', '"')
			case '\\':
				buf = append(buf, '\\', '\\')
			case '\b':
				buf = append(buf, '\\', 'b')
			case '\f':
				buf = append(buf, '\\', 'f')
			case '\n':
				buf = append(buf, '\\', 'n')
			case '\r':
				buf = append(buf, '\\', 'r')
			case '\t':
				buf = append(buf, '\\', 't')
			default:
				// 小于32的控制字符需要转义为\uXXXX
				if c < 32 {
					buf = append(buf, unicodeHex[c]...)
				} else {
					buf = append(buf, c)
				}
			}
		} else {
			buf = append(buf, c)
		}
	}
	buf = append(buf, '"', ':')
	return buf
}

// map[string]interface{} 专用编码器
type mapStringInterfaceEncoder struct{}

// 为 mapStringInterfaceEncoder 添加 appendToBytes 方法
func (e mapStringInterfaceEncoder) appendToBytes(buf []byte, src reflect.Value) ([]byte, error) {
	if src.IsNil() {
		return append(buf, nullString...), nil
	}

	mapLen := src.Len()
	if mapLen == 0 {
		return append(buf, emptyObject...), nil
	}

	// 获取所有键并排序，确保输出稳定性
	mapKeys := src.MapKeys()
	keys := make([]string, 0, len(mapKeys))

	for _, k := range mapKeys {
		keys = append(keys, k.String())
	}
	sort.Strings(keys)

	// 开始构建JSON对象
	buf = append(buf, '{')

	var err error
	// 处理第一个键值对，避免每次都检查前缀逗号
	buf = appendMapKey(buf, keys[0])
	mapVal := src.MapIndex(reflect.ValueOf(keys[0]))
	buf, err = encodeValueToBytes(buf, mapVal)
	if err != nil {
		return buf, err
	}

	// 处理剩余键值对
	for i := 1; i < len(keys); i++ {
		buf = append(buf, ',')
		buf = appendMapKey(buf, keys[i])
		mapVal = src.MapIndex(reflect.ValueOf(keys[i]))
		buf, err = encodeValueToBytes(buf, mapVal)
		if err != nil {
			return buf, err
		}
	}

	buf = append(buf, '}')
	return buf, nil
}

type mapEncoder struct {
	valueType reflect.Type
}

// 为 mapEncoder 添加 appendToBytes 方法
func (e mapEncoder) appendToBytes(buf []byte, src reflect.Value) ([]byte, error) {
	if src.IsNil() {
		return append(buf, nullString...), nil
	}

	mapLen := src.Len()
	if mapLen == 0 {
		return append(buf, emptyObject...), nil
	}

	// 获取所有键并排序，确保输出稳定性
	mapKeys := src.MapKeys()
	keys := make([]string, 0, len(mapKeys))

	for _, k := range mapKeys {
		keys = append(keys, k.String())
	}
	sort.Strings(keys)

	// 获取值编码器
	valueEncoder := getEncoder(e.valueType)

	// 开始构建JSON对象
	buf = append(buf, '{')

	var err error
	// 处理第一个键值对，避免每次都检查前缀逗号
	buf = appendMapKey(buf, keys[0])
	mapVal := src.MapIndex(reflect.ValueOf(keys[0]))
	buf, err = valueEncoder.appendToBytes(buf, mapVal)
	if err != nil {
		return buf, err
	}

	// 处理剩余键值对
	for i := 1; i < len(keys); i++ {
		buf = append(buf, ',')
		buf = appendMapKey(buf, keys[i])
		mapVal = src.MapIndex(reflect.ValueOf(keys[i]))
		buf, err = valueEncoder.appendToBytes(buf, mapVal)
		if err != nil {
			return buf, err
		}
	}

	buf = append(buf, '}')
	return buf, nil
}
