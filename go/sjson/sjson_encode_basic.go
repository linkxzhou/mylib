package sjson

import (
	"reflect"
	"strconv"
)

// 各种类型的直接编码器实现
type nullEncoder struct{}

func (e nullEncoder) appendToBytes(buf []byte, _ reflect.Value) ([]byte, error) {
	return append(buf, nullString...), nil
}

type boolEncoder struct{}

func (e boolEncoder) appendToBytes(buf []byte, src reflect.Value) ([]byte, error) {
	if src.Bool() {
		return append(buf, trueString...), nil
	}
	return append(buf, falseString...), nil
}

type intEncoder struct{}

func (e intEncoder) appendToBytes(buf []byte, src reflect.Value) ([]byte, error) {
	return strconv.AppendInt(buf, src.Int(), 10), nil
}

type uintEncoder struct{}

func (e uintEncoder) appendToBytes(buf []byte, src reflect.Value) ([]byte, error) {
	return strconv.AppendUint(buf, src.Uint(), 10), nil
}

type floatEncoder struct{}

func (e floatEncoder) appendToBytes(buf []byte, src reflect.Value) ([]byte, error) {
	f := src.Float()

	// 优化常见的整数值浮点数，减少科学计数法表示
	if f == 0 {
		return append(buf, zeroString...), nil
	}

	if f == 1 {
		return append(buf, oneString...), nil
	}

	if f == -1 {
		return append(buf, minusOneString...), nil
	}

	// 直接使用 strconv.AppendFloat 更有效率
	return strconv.AppendFloat(buf, f, 'g', -1, src.Type().Bits()), nil
}

type defaultEncoder struct{}

func (e defaultEncoder) appendToBytes(buf []byte, src reflect.Value) ([]byte, error) {
	// 默认策略：转换为字符串返回
	return stringEncoderInst.appendToBytes(buf, src)
}
