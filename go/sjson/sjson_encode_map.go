package sjson

import (
	"bytes"
	"encoding"
	"fmt"
	"reflect"
	"slices"
	"strconv"
)

// map[string]interface{} 专用编码器
type mapStringInterfaceEncoder struct{}

// 为 mapStringInterfaceEncoder 添加 appendToBytes 方法
func (e mapStringInterfaceEncoder) appendToBytes(buf []byte, src reflect.Value) ([]byte, error) {
	if src.IsNil() {
		return append(buf, nullString...), nil
	}

	mapLen := src.Len()
	if mapLen == 0 {
		return append(buf, emptyObject...), nil
	}

	// 开始构建JSON对象
	buf = append(buf, '{')

	var (
		mi  = src.MapRange()
		err error
	)

	if defaultConfig.SortMapKeys {
		sv := make([]reflectWithString, src.Len())

		for i := 0; mi.Next(); i++ {
			if sv[i].ks, err = resolveKeyName(buf, mi.Key()); err != nil {
				return nil, fmt.Errorf("json: encoding error for type %q: %q",
					src.Type().String(), err.Error())
			}
			sv[i].v = mi.Value()
		}

		slices.SortFunc(sv, func(i, j reflectWithString) int {
			return bytes.Compare(i.ks, j.ks)
		})

		for i, kv := range sv {
			if i > 0 {
				buf = append(buf, ',')
			}
			buf = append(buf, '"')
			buf = append(buf, kv.ks...)
			buf = append(buf, '"')
			buf = append(buf, ':')
			buf, err = encodeValueToBytes(buf, kv.v)
			if err != nil {
				return buf, err
			}
		}
	} else {
		for i := 0; mi.Next(); i++ {
			ks, err := resolveKeyName(buf, mi.Key())
			if err != nil {
				return nil, fmt.Errorf("json: encoding error for type %q: %q",
					src.Type().String(), err.Error())
			}

			if i > 0 {
				buf = append(buf, ',')
			}
			buf = append(buf, '"')
			buf = append(buf, ks...)
			buf = append(buf, '"')
			buf = append(buf, ':')
			buf, err = encodeValueToBytes(buf, mi.Value())
			if err != nil {
				return buf, err
			}
		}
	}

	buf = append(buf, '}')
	return buf, nil
}

type mapEncoder struct {
	valueType reflect.Type
}

// 为 mapEncoder 添加 appendToBytes 方法
func (e mapEncoder) appendToBytes(buf []byte, src reflect.Value) ([]byte, error) {
	if src.IsNil() {
		return append(buf, nullString...), nil
	}

	mapLen := src.Len()
	if mapLen == 0 {
		return append(buf, emptyObject...), nil
	}

	// 确定 value 类型，只需要获取一次 elemEncoder 即可
	elemEncoder := getEncoder(e.valueType)
	// 开始构建JSON对象
	buf = append(buf, '{')

	var (
		err error
		mi  = src.MapRange()
	)

	if defaultConfig.SortMapKeys {
		sv := make([]reflectWithString, mapLen)

		for i := 0; mi.Next(); i++ {
			if sv[i].ks, err = resolveKeyName(buf, mi.Key()); err != nil {
				return nil, fmt.Errorf("json: encoding error for type %q: %q",
					src.Type().String(), err.Error())
			}
			sv[i].v = mi.Value()
		}

		slices.SortFunc(sv, func(i, j reflectWithString) int {
			return bytes.Compare(i.ks, j.ks)
		})

		for i, kv := range sv {
			if i > 0 {
				buf = append(buf, ',')
			}
			buf = append(buf, '"')
			buf = append(buf, kv.ks...)
			buf = append(buf, '"')
			buf = append(buf, ':')
			buf, err = elemEncoder.appendToBytes(buf, kv.v)
			if err != nil {
				return buf, err
			}
		}
	} else {
		for i := 0; mi.Next(); i++ {
			ks, err := resolveKeyName(buf, mi.Key())
			if err != nil {
				return nil, fmt.Errorf("json: encoding error for type %q: %q",
					src.Type().String(), err.Error())
			}

			if i > 0 {
				buf = append(buf, ',')
			}

			buf = append(buf, '"')
			buf = append(buf, ks...)
			buf = append(buf, '"')
			buf = append(buf, ':')
			buf, err = elemEncoder.appendToBytes(buf, mi.Value())
			if err != nil {
				return buf, err
			}
		}
	}

	buf = append(buf, '}')
	return buf, nil
}

type reflectWithString struct {
	v  reflect.Value
	ks []byte
}

func resolveKeyName(buf []byte, src reflect.Value) ([]byte, error) {
	if src.Kind() == reflect.String {
		return stringToBytes(src.String()), nil
	}

	if tm, ok := src.Interface().(encoding.TextMarshaler); ok {
		if src.Kind() == reflect.Pointer && src.IsNil() {
			return emptyString, nil
		}
		buf, err := tm.MarshalText()
		return buf, err
	}

	switch src.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return strconv.AppendInt(buf, src.Int(), 10), nil
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return strconv.AppendUint(buf, src.Uint(), 10), nil
	}
	panic("unexpected map key type")
}
