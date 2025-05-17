package sjson

import (
	"reflect"
	"sync"
)

// 预定义常量字符串，减少内存分配
var (
	emptyArray  = []byte("[]")
	emptyObject = []byte("{}")
	emptyString = []byte(`""`)
	nullString  = []byte("null")
	trueString  = []byte("true")
	falseString = []byte("false")
)

// 为常用类型预分配的直接编码器
var (
	// 基本类型编码器
	boolEncoderInst      = boolEncoder{}
	intEncoderInst       = intEncoder{}
	uintEncoderInst      = uintEncoder{}
	float32EncoderInst   = float32Encoder{}
	float64EncoderInst   = float64Encoder{}
	stringEncoderInst    = stringEncoder{}
	interfaceEncoderInst = interfaceEncoder{}
	defaultEncoderInst   = defaultEncoder{}
	noSupportEncoderInst = noSupportEncoder{}
)

// Encoder 是直接编码器接口，直接将Go类型编码为JSON
type Encoder interface {
	// 新增基于字节的编码方法
	appendToBytes(*encoderStream, reflect.Value) error
}

// 直接编码器缓存
var EncoderCache sync.Map // map[reflect.Type]Encoder

// 使用小对象缓存池，避免频繁创建编码器实例
var sliceEncoderPool sync.Map
var mapEncoderPool sync.Map
var ptrEncoderPool sync.Map

// encodeValueToBytes 直接将Go值编码到字节切片中
func encodeValueToBytes(stream *encoderStream, src reflect.Value, typ reflect.Type) error {
	if !src.IsValid() {
		stream.buffer = append(stream.buffer, nullString...)
		return nil
	}

	encoder := getEncoder(typ)
	return encoder.appendToBytes(stream, src)
}

// 优化: 检测空值
//
//go:inline
func isEmptyValue(v reflect.Value) bool {
	switch v.Kind() {
	case reflect.Array, reflect.Map, reflect.Slice, reflect.String:
		return v.Len() == 0
	case reflect.Bool:
		return !v.Bool()
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return v.Int() == 0
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return v.Uint() == 0
	case reflect.Float32, reflect.Float64:
		return v.Float() == 0
	case reflect.Interface, reflect.Ptr:
		return v.IsNil()
	}
	return false
}

// 根据类型获取直接编码器
func getEncoder(t reflect.Type) Encoder {
	if t == nil {
		return nullEncoder{}
	}

	// 检查缓存
	if enc, ok := EncoderCache.Load(t); ok {
		return enc.(Encoder)
	}

	var enc Encoder

	// 使用预分配的基本类型编码器实例
	switch t.Kind() {
	case reflect.Bool:
		enc = boolEncoderInst
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		enc = intEncoderInst
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		enc = uintEncoderInst
	case reflect.Float32:
		enc = float32EncoderInst
	case reflect.Float64:
		enc = float64EncoderInst
	case reflect.String:
		enc = stringEncoderInst
	case reflect.Slice, reflect.Array:
		if t.Elem().Kind() == reflect.Uint8 {
			// []byte 特殊处理为字符串
			enc = byteSliceEncoder{}
		} else {
			// 检查是否有缓存的 sliceEncoder
			if cachedEnc, ok := sliceEncoderPool.Load(t.Elem()); ok {
				enc = cachedEnc.(Encoder)
			} else {
				sliceEnc := sliceEncoder{elemType: t.Elem()}
				sliceEncoderPool.Store(t.Elem(), sliceEnc)
				enc = sliceEnc
			}
		}
	case reflect.Map:
		// 针对 map[string]interface{} 类型优化
		switch t.Key().Kind() {
		case reflect.String,
			reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
			reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
			if t.Elem().Kind() == reflect.Interface {
				enc = mapStringInterfaceEncoder{}
			} else {
				if cachedEnc, ok := mapEncoderPool.Load(t.Elem()); ok {
					enc = cachedEnc.(Encoder)
				} else {
					mapEnc := mapEncoder{valueType: t.Elem()}
					mapEncoderPool.Store(t.Elem(), mapEnc)
					enc = mapEnc
				}
			}
		default:
			enc = noSupportEncoderInst
		}
	case reflect.Struct:
		// 结构体编码器优化，预缓存字段信息
		fields := getStructFields(t)
		enc = &structEncoder{
			typ:    t,
			fields: fields,
		}
	case reflect.Interface:
		enc = interfaceEncoderInst
	case reflect.Ptr:
		// 检查是否有缓存的 ptrEncoder
		if cachedEnc, ok := ptrEncoderPool.Load(t.Elem()); ok {
			enc = cachedEnc.(Encoder)
		} else {
			ptrEnc := ptrEncoder{elemType: t.Elem()}
			ptrEncoderPool.Store(t.Elem(), ptrEnc)
			enc = ptrEnc
		}
	default:
		enc = defaultEncoderInst
	}

	EncoderCache.Store(t, enc)
	return enc
}
