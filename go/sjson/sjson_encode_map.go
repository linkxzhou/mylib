package sjson

import (
	"bytes"
	"encoding"
	"fmt"
	"reflect"
	"slices"
)

// map[string]interface{} 专用编码器
type mapStringInterfaceEncoder struct{}

// 为 mapStringInterfaceEncoder 添加 appendToBytes 方法
func (e mapStringInterfaceEncoder) appendToBytes(stream *encoderStream, src reflect.Value) error {
	if src.IsNil() {
		stream.buffer = append(stream.buffer, nullString...)
		return nil
	}

	mapLen := src.Len()
	if mapLen == 0 {
		stream.buffer = append(stream.buffer, emptyObject...)
		return nil
	}

	// 开始构建JSON对象
	stream.buffer = append(stream.buffer, '{')

	var (
		mi  = src.MapRange()
		err error
	)

	if defaultConfig.SortMapKeys {
		sv := make([]reflectWithString, src.Len())

		for i := 0; mi.Next(); i++ {
			if sv[i].ks, err = resolveKeyName(mi.Key()); err != nil {
				return fmt.Errorf("json: encoding error for type %q: %q",
					src.Type().String(), err.Error())
			}
			sv[i].v = mi.Value()
		}

		slices.SortFunc(sv, func(i, j reflectWithString) int {
			return bytes.Compare(i.ks, j.ks)
		})

		for i, kv := range sv {
			if i > 0 {
				stream.buffer = append(stream.buffer, ',')
			}
			stream.buffer = append(stream.buffer, '"')
			stream.buffer = append(stream.buffer, kv.ks...)
			stream.buffer = append(stream.buffer, '"')
			stream.buffer = append(stream.buffer, ':')
			kvValue := kv.v

			// 获取元素的编码器
			elemEncoder := getEncoder(kvValue.Type())
			err = elemEncoder.appendToBytes(stream, kvValue)
			if err != nil {
				return err
			}
		}
	} else {
		for i := 0; mi.Next(); i++ {
			ks, err := resolveKeyName(mi.Key())
			if err != nil {
				return fmt.Errorf("json: encoding error for type %q: %q",
					src.Type().String(), err.Error())
			}

			if i > 0 {
				stream.buffer = append(stream.buffer, ',')
			}
			stream.buffer = append(stream.buffer, '"')
			stream.buffer = append(stream.buffer, ks...)
			stream.buffer = append(stream.buffer, '"')
			stream.buffer = append(stream.buffer, ':')
			miValue := mi.Value()

			// 获取元素的编码器
			elemEncoder := getEncoder(miValue.Type())
			err = elemEncoder.appendToBytes(stream, miValue)
			if err != nil {
				return err
			}
		}
	}

	stream.buffer = append(stream.buffer, '}')
	return nil
}

type mapEncoder struct {
	valueType reflect.Type
}

// 为 mapEncoder 添加 appendToBytes 方法
func (e mapEncoder) appendToBytes(stream *encoderStream, src reflect.Value) error {
	if src.IsNil() {
		stream.buffer = append(stream.buffer, nullString...)
		return nil
	}

	mapLen := src.Len()
	if mapLen == 0 {
		stream.buffer = append(stream.buffer, emptyObject...)
		return nil
	}

	// 确定 value 类型，只需要获取一次 elemEncoder 即可
	elemEncoder := getEncoder(e.valueType)
	// 开始构建JSON对象
	stream.buffer = append(stream.buffer, '{')

	var (
		err error
		mi  = src.MapRange()
	)

	if defaultConfig.SortMapKeys {
		sv := make([]reflectWithString, mapLen)

		for i := 0; mi.Next(); i++ {
			if sv[i].ks, err = resolveKeyName(mi.Key()); err != nil {
				return fmt.Errorf("json: encoding error for type %q: %q",
					src.Type().String(), err.Error())
			}
			sv[i].v = mi.Value()
		}

		slices.SortFunc(sv, func(i, j reflectWithString) int {
			return bytes.Compare(i.ks, j.ks)
		})

		for i, kv := range sv {
			if i > 0 {
				stream.buffer = append(stream.buffer, ',')
			}
			stream.buffer = append(stream.buffer, '"')
			stream.buffer = append(stream.buffer, kv.ks...)
			stream.buffer = append(stream.buffer, '"')
			stream.buffer = append(stream.buffer, ':')
			err = elemEncoder.appendToBytes(stream, kv.v)
			if err != nil {
				return err
			}
		}
	} else {
		for i := 0; mi.Next(); i++ {
			ks, err := resolveKeyName(mi.Key())
			if err != nil {
				return fmt.Errorf("json: encoding error for type %q: %q",
					src.Type().String(), err.Error())
			}

			if i > 0 {
				stream.buffer = append(stream.buffer, ',')
			}

			stream.buffer = append(stream.buffer, '"')
			stream.buffer = append(stream.buffer, ks...)
			stream.buffer = append(stream.buffer, '"')
			stream.buffer = append(stream.buffer, ':')
			err = elemEncoder.appendToBytes(stream, mi.Value())
			if err != nil {
				return err
			}
		}
	}

	stream.buffer = append(stream.buffer, '}')
	return nil
}

type reflectWithString struct {
	v  reflect.Value
	ks []byte
}

//go:inline
func resolveKeyName(src reflect.Value) ([]byte, error) {
	if src.Kind() == reflect.String {
		return stringToBytes(src.String()), nil
	}

	if tm, ok := src.Interface().(encoding.TextMarshaler); ok {
		if src.Kind() == reflect.Pointer && src.IsNil() {
			return emptyString, nil
		}
		return tm.MarshalText()
	}

	switch src.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return appendInt(nil, src.Int(), 10), nil
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return appendUint(nil, src.Uint(), 10), nil
	}

	return nil, fmt.Errorf("unexpected map key type: %v", src.Type())
}
