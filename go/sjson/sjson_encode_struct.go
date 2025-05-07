package sjson

import (
	"reflect"
)

// structField 表示结构体字段的缓存信息
type structField struct {
	name      string
	index     int
	omitempty bool
}

type structEncoder struct {
	typ    reflect.Type
	fields []structField
}

// 添加appendToBytes方法，将结构体直接编码到字节切片
func (e *structEncoder) appendToBytes(buf []byte, src reflect.Value) ([]byte, error) {
	if src.Kind() == reflect.Ptr {
		if src.IsNil() {
			return append(buf, nullString...), nil
		}
		src = src.Elem()
	}

	// 开始对象
	buf = append(buf, '{')

	// 跟踪是否有字段被编码
	first := true
	var err error

	// 逐个编码字段
	for _, field := range e.fields {
		// 获取字段值
		f := src.Field(field.index)

		// 处理omitempty标签
		if field.omitempty && isEmptyValue(f) {
			continue
		}

		// 添加逗号分隔符（非第一个字段）
		if first {
			first = false
		} else {
			buf = append(buf, ',')
		}

		// 写入字段名（使用之前实现的appendMapKey函数）
		buf = appendMapKey(buf, field.name)

		// 编码字段值
		buf, err = encodeValueToBytes(buf, f)
		if err != nil {
			return buf, err
		}
	}

	// 结束对象
	if first {
		// 如果没有字段被编码，表示是一个空对象
		// 不能像strings.Builder那样重置，需要替换整个内容
		return append(buf[:0], emptyObject...), nil
	} else {
		buf = append(buf, '}')
	}

	return buf, nil
}
