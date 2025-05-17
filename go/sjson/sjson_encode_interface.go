package sjson

import (
	"reflect"
)

type sliceEncoder struct {
	elemType reflect.Type
}

func (e sliceEncoder) appendToBytes(stream *encoderStream, src reflect.Value) error {
	if src.IsNil() {
		stream.buffer = append(stream.buffer, nullString...)
		return nil
	}

	length := src.Len()
	if length == 0 {
		stream.buffer = append(stream.buffer, emptyArray...)
		return nil
	}

	stream.buffer = append(stream.buffer, '[')

	// 获取元素的编码器
	elemEncoder := getEncoder(e.elemType)

	var err error

	// 编码剩余元素
	for i := 0; i < length; i++ {
		if i > 0 {
			stream.buffer = append(stream.buffer, ',')
		}
		err = elemEncoder.appendToBytes(stream, src.Index(i))
		if err != nil {
			return err
		}
	}

	stream.buffer = append(stream.buffer, ']')
	return nil
}

type interfaceEncoder struct{}

func (e interfaceEncoder) appendToBytes(stream *encoderStream, src reflect.Value) error {
	if src.IsNil() {
		stream.buffer = append(stream.buffer, nullString...)
		return nil
	}

	// 获取接口中实际的值
	elem := src.Elem()

	// 获取元素的编码器
	elemEncoder := getEncoder(elem.Type())
	return elemEncoder.appendToBytes(stream, elem)
}

type ptrEncoder struct {
	elemType reflect.Type
}

func (e ptrEncoder) appendToBytes(stream *encoderStream, src reflect.Value) error {
	if src.IsNil() {
		stream.buffer = append(stream.buffer, nullString...)
		return nil
	}

	// 获取指针指向的值
	elemVal := src.Elem()

	// 使用预先缓存的元素编码器
	elemEncoder := getEncoder(e.elemType)
	return elemEncoder.appendToBytes(stream, elemVal)
}
