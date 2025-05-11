package sjson

import (
	"reflect"
)

// structField 表示结构体字段的缓存信息
type structField struct {
	name      []byte
	index     int
	omitempty bool
	typ       reflect.Type
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

	var err error

	// 逐个编码字段
	for i, field := range e.fields {
		// 获取字段值
		f := src.Field(field.index)

		// 处理omitempty标签
		if field.omitempty && isEmptyValue(f) {
			continue
		}

		// 添加逗号分隔符（非第一个字段）
		if i > 0 {
			buf = append(buf, ',')
		}

		// 写入字段名（使用之前实现的appendMapKey函数）
		buf = append(buf, '"')
		buf = append(buf, field.name...)
		buf = append(buf, '"', ':')

		elemEncoder := getEncoder(field.typ)
		// 编码字段值
		buf, err = elemEncoder.appendToBytes(buf, f)
		if err != nil {
			return buf, err
		}
	}

	buf = append(buf, '}')
	return buf, nil
}
