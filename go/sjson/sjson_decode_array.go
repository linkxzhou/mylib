package sjson

import (
	"fmt"
	"reflect"
)

// 解码数组
func (d *Decoder) decodeArray(dst reflect.Value) error {
	// 跳过左方括号
	d.nextToken()

	kind := dst.Kind()

	// 空数组快速路径
	if d.token.Type == RightBracketToken {
		d.nextToken() // 跳过右方括号

		switch kind {
		case reflect.Slice:
			// 创建空切片
			dst.Set(reflect.MakeSlice(dst.Type(), 0, 0))
		case reflect.Interface:
			if dst.NumMethod() == 0 {
				// 创建空数组并设置到接口
				dst.Set(reflect.ValueOf([]interface{}{}))
			}
		}
		return nil
	}

	switch kind {
	case reflect.Slice:
		return d.decodeSlice(dst)
	case reflect.Array:
		return d.decodeFixedArray(dst)
	case reflect.Interface:
		if dst.NumMethod() == 0 {
			return d.decodeInterfaceArray(dst)
		}
	}

	// 跳过整个数组
	var depth int = 1 // 已经读取了一个左方括号
	for depth > 0 && d.token.Type != EOFToken {
		if d.token.Type == LeftBracketToken {
			depth++
		} else if d.token.Type == RightBracketToken {
			depth--
		}
		d.nextToken()
	}

	return fmt.Errorf("无法将数组解码到 %s 类型", dst.Type())
}

// 解码到切片
func (d *Decoder) decodeSlice(dst reflect.Value) error {
	// 从对象池获取切片
	elemValues := valueSlicePool.Get().(*[]reflect.Value)
	*elemValues = (*elemValues)[:0] // 清空但保留容量

	// 确保函数结束时归还切片到池
	defer valueSlicePool.Put(elemValues)

	elemType := dst.Type().Elem()

	// 收集元素
	for {
		// 解码值
		elem := reflect.New(elemType).Elem()
		if err := d.decodeValue(elem); err != nil {
			return err
		}
		*elemValues = append(*elemValues, elem)

		// 检查分隔符
		if d.token.Type == CommaToken {
			d.nextToken()
		} else if d.token.Type == RightBracketToken {
			d.nextToken() // 跳过右方括号
			break
		} else {
			return fmt.Errorf("数组中意外的标记: %v", d.token)
		}
	}

	// 创建最终切片
	length := len(*elemValues)
	sliceValue := reflect.MakeSlice(dst.Type(), length, length)

	// 复制元素到最终切片
	for i, elem := range *elemValues {
		sliceValue.Index(i).Set(elem)
	}

	// 设置结果
	dst.Set(sliceValue)
	return nil
}

// 解码到固定数组
func (d *Decoder) decodeFixedArray(dst reflect.Value) error {
	// 不需要额外的内存分配，直接解码到目标数组
	arrayLen := dst.Len()

	for i := 0; i < arrayLen; i++ {
		// 如果JSON数组结束，跳出循环
		if d.token.Type == RightBracketToken {
			d.nextToken() // 跳过右方括号
			break
		}

		// 解码到数组元素
		if err := d.decodeValue(dst.Index(i)); err != nil {
			return err
		}

		// 检查分隔符
		if d.token.Type == CommaToken {
			d.nextToken()
		} else if d.token.Type == RightBracketToken {
			d.nextToken() // 跳过右方括号
			break
		} else {
			return fmt.Errorf("数组中意外的标记: %v", d.token)
		}
	}

	// 如果JSON数组元素多于Go数组长度，跳过多余元素
	for d.token.Type != RightBracketToken && d.token.Type != EOFToken {
		if err := d.skipValue(); err != nil {
			return err
		}

		// 检查分隔符
		if d.token.Type == CommaToken {
			d.nextToken()
		} else if d.token.Type == RightBracketToken {
			d.nextToken() // 跳过右方括号
			break
		} else {
			return fmt.Errorf("数组中意外的标记: %v", d.token)
		}
	}

	return nil
}

// 解码到接口数组
func (d *Decoder) decodeInterfaceArray(dst reflect.Value) error {
	// 从对象池获取接口切片
	elements := interfaceSlicePool.Get().(*[]interface{})
	*elements = (*elements)[:0] // 清空但保留容量

	// 确保函数结束时归还切片到池
	defer interfaceSlicePool.Put(elements)

	// 解析元素
	for {
		// 解码到临时接口
		var element interface{}
		elemValue := reflect.ValueOf(&element).Elem()

		if err := d.decodeValue(elemValue); err != nil {
			return err
		}

		*elements = append(*elements, element)

		// 检查分隔符
		if d.token.Type == CommaToken {
			d.nextToken()
		} else if d.token.Type == RightBracketToken {
			d.nextToken() // 跳过右方括号
			break
		} else {
			return fmt.Errorf("数组中意外的标记: %v", d.token)
		}
	}

	// 设置结果
	dst.Set(reflect.ValueOf(*elements))
	return nil
}
