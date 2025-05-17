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
func (e *structEncoder) appendToBytes(stream *encoderStream, src reflect.Value) error {
	if src.Kind() == reflect.Ptr {
		if src.IsNil() {
			stream.buffer = append(stream.buffer, nullString...)
			return nil
		}
		src = src.Elem()
	}

	// 开始对象
	stream.buffer = append(stream.buffer, '{')

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
			stream.buffer = append(stream.buffer, ',')
		}

		// 写入字段名（使用之前实现的appendMapKey函数）
		stream.buffer = append(stream.buffer, '"')
		stream.buffer = append(stream.buffer, field.name...)
		stream.buffer = append(stream.buffer, '"', ':')

		elemEncoder := getEncoder(field.typ)
		// 编码字段值
		err = elemEncoder.appendToBytes(stream, f)
		if err != nil {
			return err
		}
	}

	stream.buffer = append(stream.buffer, '}')
	return nil
}
