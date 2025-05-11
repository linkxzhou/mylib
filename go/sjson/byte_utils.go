package sjson

import (
	"errors"
	"math"
	"reflect"
	"unsafe"
)

// 直接从字节切片解析整数，避免string转换
func parseIntFromBytes(b []byte, base int, bitSize int) (int64, error) {
	if len(b) == 0 {
		return 0, errors.New("空字节切片")
	}

	// 处理符号
	var negative bool
	var i int
	if b[0] == '+' {
		i = 1
	} else if b[0] == '-' {
		negative = true
		i = 1
	}

	// 检查是否有数字
	if i >= len(b) {
		return 0, errors.New("无效的数字格式")
	}

	// 计算值
	var n int64
	for ; i < len(b); i++ {
		var v byte
		d := b[i]

		switch {
		case '0' <= d && d <= '9':
			v = d - '0'
		case 'a' <= d && d <= 'z':
			v = d - 'a' + 10
		case 'A' <= d && d <= 'Z':
			v = d - 'A' + 10
		default:
			return 0, errors.New("无效的数字字符")
		}

		if int(v) >= base {
			return 0, errors.New("数字超出进制范围")
		}

		// 检查溢出
		if n > math.MaxInt64/int64(base) {
			// 溢出
			if negative {
				return math.MinInt64, nil
			}
			return math.MaxInt64, nil
		}

		n *= int64(base)
		n += int64(v)
	}

	if negative {
		n = -n
	}

	// 根据bitSize检查范围
	switch bitSize {
	case 8:
		if n < math.MinInt8 || n > math.MaxInt8 {
			return 0, errors.New("数值超出int8范围")
		}
	case 16:
		if n < math.MinInt16 || n > math.MaxInt16 {
			return 0, errors.New("数值超出int16范围")
		}
	case 32:
		if n < math.MinInt32 || n > math.MaxInt32 {
			return 0, errors.New("数值超出int32范围")
		}
	}

	return n, nil
}

// 直接从字节切片解析无符号整数，避免string转换
func parseUintFromBytes(b []byte, base int, bitSize int) (uint64, error) {
	if len(b) == 0 {
		return 0, errors.New("空字节切片")
	}

	// 处理符号
	var i int
	if b[0] == '+' {
		i = 1
	}

	// 检查是否有数字
	if i >= len(b) {
		return 0, errors.New("无效的数字格式")
	}

	// 计算值
	var n uint64
	for ; i < len(b); i++ {
		var v byte
		d := b[i]

		switch {
		case '0' <= d && d <= '9':
			v = d - '0'
		case 'a' <= d && d <= 'z':
			v = d - 'a' + 10
		case 'A' <= d && d <= 'Z':
			v = d - 'A' + 10
		default:
			return 0, errors.New("无效的数字字符")
		}

		if int(v) >= base {
			return 0, errors.New("数字超出进制范围")
		}

		// 检查溢出
		if n > math.MaxUint64/uint64(base) {
			// 溢出
			return math.MaxUint64, nil
		}

		n *= uint64(base)
		n += uint64(v)
	}

	// 根据bitSize检查范围
	switch bitSize {
	case 8:
		if n > math.MaxUint8 {
			return 0, errors.New("数值超出uint8范围")
		}
	case 16:
		if n > math.MaxUint16 {
			return 0, errors.New("数值超出uint16范围")
		}
	case 32:
		if n > math.MaxUint32 {
			return 0, errors.New("数值超出uint32范围")
		}
	}

	return n, nil
}

// 直接从字节切片解析浮点数，避免string转换
func parseFloatFromBytes(b []byte, bitSize int) (float64, error) {
	if len(b) == 0 {
		return 0, errors.New("空字节切片")
	}

	// 处理符号
	var negative bool
	var i int
	if b[0] == '+' {
		i = 1
	} else if b[0] == '-' {
		negative = true
		i = 1
	}

	// 解析整数部分
	var n float64
	var sawDigit bool

	for ; i < len(b); i++ {
		if b[i] == '.' {
			i++
			break
		}
		if b[i] == 'e' || b[i] == 'E' {
			break
		}
		if '0' <= b[i] && b[i] <= '9' {
			sawDigit = true
			n = n*10 + float64(b[i]-'0')
		} else {
			return 0, errors.New("无效的数字字符")
		}
	}

	// 解析小数部分
	if i < len(b) && b[i-1] == '.' {
		decimal := 0.1
		for ; i < len(b); i++ {
			if b[i] == 'e' || b[i] == 'E' {
				break
			}
			if '0' <= b[i] && b[i] <= '9' {
				sawDigit = true
				n += decimal * float64(b[i]-'0')
				decimal *= 0.1
			} else {
				return 0, errors.New("无效的数字字符")
			}
		}
	}

	if !sawDigit {
		return 0, errors.New("无效的数字格式")
	}

	// 处理指数部分
	if i < len(b) && (b[i] == 'e' || b[i] == 'E') {
		i++
		if i >= len(b) {
			return 0, errors.New("无效的指数格式")
		}

		expSign := 1
		if b[i] == '+' {
			i++
		} else if b[i] == '-' {
			expSign = -1
			i++
		}

		if i >= len(b) || b[i] < '0' || b[i] > '9' {
			return 0, errors.New("无效的指数格式")
		}

		var exp int
		for ; i < len(b); i++ {
			if '0' <= b[i] && b[i] <= '9' {
				exp = exp*10 + int(b[i]-'0')
			} else {
				return 0, errors.New("无效的指数字符")
			}
		}

		// 应用指数
		if expSign > 0 {
			for j := 0; j < exp; j++ {
				n *= 10
			}
		} else {
			for j := 0; j < exp; j++ {
				n /= 10
			}
		}
	}

	if negative {
		n = -n
	}

	// 根据bitSize检查范围
	if bitSize == 32 {
		// 直接转换为float32再转回float64，不做额外的范围检查
		// 如果值超出范围，Go 会自动处理为 Inf
		return float64(float32(n)), nil
	}

	return n, nil
}

// stringToBytes 将 string 转换为 []byte，零拷贝（不安全）
func stringToBytes(s string) []byte {
	sh := (*reflect.StringHeader)(unsafe.Pointer(&s))
	slice := reflect.SliceHeader{
		Data: sh.Data,
		Len:  sh.Len,
		Cap:  sh.Len,
	}
	return *(*[]byte)(unsafe.Pointer(&slice))
}

// bytesToString 将 []byte 转换为 string，零拷贝（不安全）
func bytesToString(b []byte) string {
	return *(*string)(unsafe.Pointer(&b))
}
