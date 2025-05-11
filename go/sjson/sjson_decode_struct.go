package sjson

import (
	"fmt"
	"reflect"
	"sync"
)

// 解码对象
func (d *Decoder) decodeObject(dst reflect.Value) error {
	// 跳过左大括号
	d.nextToken()

	kind := dst.Kind()

	// 空对象快速路径
	if d.token.Type == RightBraceToken {
		d.nextToken() // 跳过右大括号

		switch kind {
		case reflect.Map:
			// 如果是空map，创建一个空map
			if dst.IsNil() {
				dst.Set(reflect.MakeMap(dst.Type()))
			}
			return nil

		case reflect.Interface:
			if dst.NumMethod() == 0 {
				dst.Set(reflect.MakeMap(reflect.TypeOf(map[string]interface{}{})))
				return nil
			}
		}

		// 其他类型，空对象被忽略
		return nil
	}

	switch kind {
	case reflect.Map:
		return d.decodeMap(dst)
	case reflect.Struct:
		return d.decodeStruct(dst)
	case reflect.Interface:
		if dst.NumMethod() == 0 {
			emptyMapValue := reflect.MakeMap(reflect.TypeOf(map[string]interface{}{}))

			// 解码到这个map
			if err := d.decodeMap(emptyMapValue); err != nil {
				return err
			}

			// 设置到接口值
			dst.Set(emptyMapValue)
			return nil
		}
	}

	// 跳过整个对象
	var depth int = 1 // 已经读取了一个左大括号
	for depth > 0 && d.token.Type != EOFToken {
		if d.token.Type == LeftBraceToken {
			depth++
		} else if d.token.Type == RightBraceToken {
			depth--
		}
		d.nextToken()
	}

	return fmt.Errorf("无法将对象解码到 %s 类型", dst.Type())
}

// 解码Map
func (d *Decoder) decodeMap(dst reflect.Value) error {
	if dst.IsNil() {
		dst.Set(reflect.MakeMap(dst.Type()))
	}

	elemType := dst.Type().Elem()

	for {
		// 键必须是字符串
		if d.token.Type != StringToken {
			return fmt.Errorf("对象键必须是字符串，得到: %v", d.token)
		}

		key := d.token.Value
		d.nextToken()

		// 键后面必须是冒号
		if d.token.Type != ColonToken {
			return fmt.Errorf("对象键后面必须是冒号，得到: %v", d.token)
		}
		d.nextToken()

		// 解码值
		valueElem := reflect.New(elemType).Elem()
		if err := d.decodeValue(valueElem); err != nil {
			return err
		}

		// 设置键值对，确保键是字符串类型
		keyValue := reflect.ValueOf(bytesToString(key))
		dst.SetMapIndex(keyValue, valueElem)

		// 检查是否有更多的键值对
		if d.token.Type == CommaToken {
			d.nextToken()
		} else if d.token.Type == RightBraceToken {
			d.nextToken() // 跳过右大括号
			break
		} else {
			return fmt.Errorf("对象中意外的标记: %v", d.token)
		}
	}

	return nil
}

// 缓存结构体字段映射
var fieldMapCache = sync.Map{} // map[reflect.Type]map[string]int

// 获取字段映射（带缓存）
func getFieldMap(structType reflect.Type, fields []structField) map[string]int {
	// 检查缓存
	if cachedMap, ok := fieldMapCache.Load(structType); ok {
		return cachedMap.(map[string]int)
	}

	// 创建新的映射
	fieldMap := make(map[string]int, len(fields))
	for _, field := range fields {
		fieldMap[bytesToString(field.name)] = field.index
	}

	// 存入缓存
	fieldMapCache.Store(structType, fieldMap)
	return fieldMap
}

// 解码结构体
func (d *Decoder) decodeStruct(dst reflect.Value) error {
	structType := dst.Type()

	// 预先获取所有字段信息，避免重复查找
	fields := getStructFields(structType)

	// 获取字段映射（使用缓存）
	fieldMap := getFieldMap(structType, fields)

	for {
		// 键必须是字符串
		if d.token.Type != StringToken {
			return fmt.Errorf("对象键必须是字符串，得到: %v", d.token)
		}

		key := d.token.Value
		d.nextToken()

		// 键后面必须是冒号
		if d.token.Type != ColonToken {
			return fmt.Errorf("对象键后面必须是冒号，得到: %v", d.token)
		}
		d.nextToken()

		// 查找结构体字段
		fieldIndex, exists := fieldMap[bytesToString(key)]

		if exists && fieldIndex >= 0 {
			// 字段存在，解码值
			field := dst.Field(fieldIndex)
			if field.CanSet() {
				if err := d.decodeValue(field); err != nil {
					return fmt.Errorf("解码字段 %s 出错: %w", key, err)
				}
			} else {
				// 字段不可设置，跳过值
				if err := d.skipValue(); err != nil {
					return err
				}
			}
		} else {
			// 字段不存在，跳过值
			if err := d.skipValue(); err != nil {
				return err
			}
		}

		// 检查是否有更多的键值对
		if d.token.Type == CommaToken {
			d.nextToken()
		} else if d.token.Type == RightBraceToken {
			d.nextToken() // 跳过右大括号
			break
		} else {
			return fmt.Errorf("对象中意外的标记: %v", d.token)
		}
	}

	return nil
}
