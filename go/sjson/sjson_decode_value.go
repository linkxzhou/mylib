package sjson

import (
	"fmt"
	"reflect"
	"strconv"
)

// 解码任意值到目标反射值
func (d *Decoder) decodeValue(dst reflect.Value) error {
	if !dst.IsValid() {
		return fmt.Errorf("解码目标无效")
	}

	// 特殊处理指针类型
	if dst.Kind() == reflect.Ptr {
		// 如果是nil指针，需要先初始化
		if dst.IsNil() {
			dst.Set(reflect.New(dst.Type().Elem()))
		}
		// 递归解码指针指向的值
		return d.decodeValue(dst.Elem())
	}

	// 目标必须可设置
	if !dst.CanSet() {
		return fmt.Errorf("解码目标不可设置")
	}

	// 使用一个switch语句而不是多个if-else来提高性能
	switch d.token.Type {
	case NullToken:
		d.nextToken()
		// 对于null值，设置为零值
		dst.Set(reflect.Zero(dst.Type()))
		return nil

	case TrueToken:
		d.nextToken()
		return d.decodeBool(true, dst)

	case FalseToken:
		d.nextToken()
		return d.decodeBool(false, dst)

	case NumberToken:
		value := d.token.Value
		d.nextToken()
		return d.decodeNumber(value, dst)

	case StringToken:
		value := d.token.Value
		d.nextToken()
		return d.decodeString(value, dst)

	case LeftBraceToken:
		return d.decodeObject(dst)

	case LeftBracketToken:
		return d.decodeArray(dst)

	default:
		return fmt.Errorf("无法识别的JSON token类型: %v", d.token.Type)
	}
}

// 解码布尔值
func (d *Decoder) decodeBool(value bool, dst reflect.Value) error {
	// 直接根据Kind处理，避免多次分支判断
	kind := dst.Kind()

	// 使用直接类型判断而非switch来减少分支
	if kind == reflect.Bool {
		dst.SetBool(value)
		return nil
	}

	if kind == reflect.Interface && dst.NumMethod() == 0 {
		dst.Set(reflect.ValueOf(value))
		return nil
	}

	return fmt.Errorf("无法将布尔值解码到 %s 类型", dst.Type())
}

// 解码数字
func (d *Decoder) decodeNumber(value string, dst reflect.Value) error {
	// 快速判断整数
	isInt := true
	for i := 0; i < len(value); i++ {
		c := value[i]
		if c == '.' || c == 'e' || c == 'E' {
			isInt = false
			break
		}
	}

	// 根据当前值的类型优化解析路径
	kind := dst.Kind()

	// 整数路径
	if isInt {
		// 整数优化路径
		if kind >= reflect.Int && kind <= reflect.Int64 {
			n, err := strconv.ParseInt(value, 10, 64)
			if err != nil {
				return fmt.Errorf("无法将 %s 解析为整数: %w", value, err)
			}
			dst.SetInt(n)
			return nil
		}

		if kind >= reflect.Uint && kind <= reflect.Uint64 {
			n, err := strconv.ParseUint(value, 10, 64)
			if err != nil {
				return fmt.Errorf("无法将 %s 解析为无符号整数: %w", value, err)
			}
			dst.SetUint(n)
			return nil
		}
	}

	// 浮点数路径
	if kind == reflect.Float32 || kind == reflect.Float64 {
		n, err := strconv.ParseFloat(value, 64)
		if err != nil {
			return fmt.Errorf("无法将 %s 解析为浮点数: %w", value, err)
		}
		dst.SetFloat(n)
		return nil
	}

	// 接口路径
	if kind == reflect.Interface && dst.NumMethod() == 0 {
		// 对于空接口，始终使用float64表示数字，以保持与原始实现的兼容性
		n, err := strconv.ParseFloat(value, 64)
		if err != nil {
			return fmt.Errorf("无法将 %s 解析为数字: %w", value, err)
		}
		dst.Set(reflect.ValueOf(n))
		return nil
	}

	return fmt.Errorf("无法将数字解码到 %s 类型", dst.Type())
}

// 解码字符串
func (d *Decoder) decodeString(value string, dst reflect.Value) error {
	kind := dst.Kind()

	if kind == reflect.String {
		dst.SetString(value)
		return nil
	}

	if kind == reflect.Interface && dst.NumMethod() == 0 {
		dst.Set(reflect.ValueOf(value))
		return nil
	}

	return fmt.Errorf("无法将字符串解码到 %s 类型", dst.Type())
}
