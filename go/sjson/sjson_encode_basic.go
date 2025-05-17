package sjson

import (
	"fmt"
	"reflect"
	"strconv"
)

// 各种类型的直接编码器实现
type nullEncoder struct{}

//go:inline
//go:nosplit
func (e nullEncoder) appendToBytes(stream *encoderStream, _ reflect.Value) error {
	stream.buffer = append(stream.buffer, nullString...)
	return nil
}

type boolEncoder struct{}

//go:inline
//go:nosplit
func (e boolEncoder) appendToBytes(stream *encoderStream, src reflect.Value) error {
	if src.Bool() {
		stream.buffer = append(stream.buffer, trueString...)
	} else {
		stream.buffer = append(stream.buffer, falseString...)
	}
	return nil
}

type intEncoder struct{}

//go:inline
//go:nosplit
func (e intEncoder) appendToBytes(stream *encoderStream, src reflect.Value) error {
	intValue := src.Int()
	if intValue < 0 {
		stream.buffer = append(stream.buffer, '-')
		stream.buffer = appendInt(stream.buffer, int64(intValue), 10)
	} else {
		stream.buffer = appendInt(stream.buffer, intValue, 10)
	}

	return nil
}

type uintEncoder struct{}

//go:inline
//go:nosplit
func (e uintEncoder) appendToBytes(stream *encoderStream, src reflect.Value) error {
	stream.buffer = appendUint(stream.buffer, src.Uint(), 10)
	return nil
}

type float32Encoder struct{}

func (e float32Encoder) appendToBytes(stream *encoderStream, src reflect.Value) error {
	stream.buffer = strconv.AppendFloat(stream.buffer, src.Float(), 'f', -1, 64)
	return nil
}

type float64Encoder struct{}

func (e float64Encoder) appendToBytes(stream *encoderStream, src reflect.Value) error {
	stream.buffer = strconv.AppendFloat(stream.buffer, src.Float(), 'f', -1, 32)
	return nil
}

type defaultEncoder struct{}

func (e defaultEncoder) appendToBytes(stream *encoderStream, src reflect.Value) error {
	// 默认策略：转换为字符串返回
	return stringEncoderInst.appendToBytes(stream, src)
}

type noSupportEncoder struct{}

func (e noSupportEncoder) appendToBytes(stream *encoderStream, src reflect.Value) error {
	return fmt.Errorf("unsupported map key type: %v", src.Type())
}
