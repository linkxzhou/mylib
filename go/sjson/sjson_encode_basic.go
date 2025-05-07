package sjson

import (
	"reflect"
	"strconv"
	"unicode/utf8"
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
	i := src.Int()
	// 使用缓存的数字字符串
	if i >= -numberCacheZero && i < numberCacheSize-numberCacheZero {
		return append(buf, numberCache[i+numberCacheZero]...), nil
	}
	return strconv.AppendInt(buf, i, 10), nil
}

type uintEncoder struct{}

func (e uintEncoder) appendToBytes(buf []byte, src reflect.Value) ([]byte, error) {
	i := src.Uint()
	// 使用缓存的数字字符串
	if i < uint64(numberCacheSize-numberCacheZero) {
		return append(buf, numberCache[i+numberCacheZero]...), nil
	}
	return strconv.AppendUint(buf, i, 10), nil
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

type runeEncoder struct{}

func (e runeEncoder) appendToBytes(buf []byte, src reflect.Value) ([]byte, error) {
	r := rune(src.Int())
	return encodeRuneToBytes(buf, r)
}

// encodeRuneToBytes 将单个rune编码为JSON字符串并添加到字节切片
func encodeRuneToBytes(buf []byte, r rune) ([]byte, error) {
	tempBuf := make([]byte, 4) // UTF-8最多4字节
	n := utf8.EncodeRune(tempBuf, r)

	buf = append(buf, '"')
	for i := 0; i < n; i++ {
		// 转义特殊字符
		if tempBuf[i] < utf8.RuneSelf && safeSet[tempBuf[i]] {
			buf = append(buf, tempBuf[i])
		} else if tempBuf[i] < utf8.RuneSelf {
			switch tempBuf[i] {
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
				buf = append(buf, unicodeHex[tempBuf[i]]...)
			}
		} else {
			buf = append(buf, tempBuf[i])
		}
	}
	buf = append(buf, '"')
	return buf, nil
}

type defaultEncoder struct{}

func (e defaultEncoder) appendToBytes(buf []byte, src reflect.Value) ([]byte, error) {
	// 默认策略：转换为字符串返回
	return stringEncoderInst.appendToBytes(buf, reflect.ValueOf(src.String()))
}
