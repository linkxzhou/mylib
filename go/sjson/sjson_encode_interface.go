package sjson

import (
	"reflect"
)

type sliceEncoder struct {
	elemType reflect.Type
}

func (e sliceEncoder) appendToBytes(buf []byte, src reflect.Value) ([]byte, error) {
	if src.IsNil() {
		return append(buf, nullString...), nil
	}

	length := src.Len()
	if length == 0 {
		return append(buf, emptyArray...), nil
	}

	buf = append(buf, '[')

	// 获取元素的编码器
	elemEncoder := getEncoder(e.elemType)

	var err error
	// 第一个元素单独处理，避免每次循环都检查前缀逗号
	buf, err = elemEncoder.appendToBytes(buf, src.Index(0))
	if err != nil {
		return buf, err
	}

	// 编码剩余元素
	for i := 1; i < length; i++ {
		buf = append(buf, ',')
		buf, err = elemEncoder.appendToBytes(buf, src.Index(i))
		if err != nil {
			return buf, err
		}
	}

	buf = append(buf, ']')
	return buf, nil
}

type interfaceEncoder struct{}

func (e interfaceEncoder) appendToBytes(buf []byte, src reflect.Value) ([]byte, error) {
	if src.IsNil() {
		return append(buf, nullString...), nil
	}

	// 提取接口中的具体值
	elem := src.Elem()

	// 直接调用encodeValueToBytes，它会为elem找到合适的编码器
	return encodeValueToBytes(buf, elem)
}

type ptrEncoder struct {
	elemType reflect.Type
}

func (e ptrEncoder) appendToBytes(buf []byte, src reflect.Value) ([]byte, error) {
	if src.IsNil() {
		return append(buf, nullString...), nil
	}

	// 获取指针指向的值
	elemVal := src.Elem()

	// 使用预先缓存的元素编码器
	elemEncoder := getEncoder(e.elemType)
	return elemEncoder.appendToBytes(buf, elemVal)
}
