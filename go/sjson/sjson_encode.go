package sjson

import (
	"reflect"
	"strconv"
	"sync"
)

// 预定义常量字符串，减少内存分配
const (
	emptyArray     = "[]"
	emptyObject    = "{}"
	emptyString    = `""`
	nullString     = "null"
	trueString     = "true"
	falseString    = "false"
	zeroString     = "0"
	oneString      = "1"
	minusOneString = "-1"
)

// 常用数字缓存，最多支持[-512, 512]的整数快速转换
const (
	numberCacheSize = 1024
	numberCacheZero = 512 // 零点偏移
)

// 为常用类型预分配的直接编码器
var (
	// 基本类型编码器
	boolEncoderInst      = boolEncoder{}
	intEncoderInst       = intEncoder{}
	uintEncoderInst      = uintEncoder{}
	floatEncoderInst     = floatEncoder{}
	stringEncoderInst    = stringEncoder{}
	interfaceEncoderInst = interfaceEncoder{}
	defaultEncoderInst   = defaultEncoder{}
	runeEncoderInst      = runeEncoder{}
)

var numberCache [numberCacheSize]string

// 初始化数字缓存
func init() {
	// 为常用的整数预生成字符串
	for i := 0; i < numberCacheSize; i++ {
		numberCache[i] = strconv.FormatInt(int64(i-numberCacheZero), 10)
	}
}

// Encoder 是直接编码器接口，直接将Go类型编码为JSON
type Encoder interface {
	// 新增基于字节的编码方法
	appendToBytes([]byte, reflect.Value) ([]byte, error)
}

// 直接编码器缓存
var EncoderCache sync.Map // map[reflect.Type]Encoder

// 使用小对象缓存池，避免频繁创建编码器实例
var sliceEncoderPool sync.Map // map[reflect.Type]Encoder
var mapEncoderPool sync.Map   // map[reflect.Type]Encoder
var ptrEncoderPool sync.Map   // map[reflect.Type]Encoder

// encodeValueToBytes 直接将Go值编码到字节切片中
func encodeValueToBytes(buf []byte, src reflect.Value) ([]byte, error) {
	if !src.IsValid() {
		return append(buf, nullString...), nil
	}

	encoder := getEncoder(src.Type())
	return encoder.appendToBytes(buf, src)
}

// 优化: 检测空值
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
		// 特殊处理 int32 类型的 rune
		if t.Kind() == reflect.Int32 && t == reflect.TypeOf(rune(0)) {
			enc = runeEncoderInst
		} else {
			enc = intEncoderInst
		}
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		enc = uintEncoderInst
	case reflect.Float32, reflect.Float64:
		enc = floatEncoderInst
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
		if t.Key().Kind() == reflect.String {
			if t.Elem().Kind() == reflect.Interface {
				enc = mapStringInterfaceEncoder{}
			} else {
				// 检查是否有缓存的 mapEncoder
				if cachedEnc, ok := mapEncoderPool.Load(t.Elem()); ok {
					enc = cachedEnc.(Encoder)
				} else {
					mapEnc := mapEncoder{valueType: t.Elem()}
					mapEncoderPool.Store(t.Elem(), mapEnc)
					enc = mapEnc
				}
			}
		} else {
			// 非字符串键的 map 暂不支持，使用默认编码器
			enc = defaultEncoderInst
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
