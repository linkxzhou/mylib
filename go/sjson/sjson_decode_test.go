package sjson

import (
	"encoding/json"
	"testing"
)

// 用于测试直接解码器的结构体
type DecodeTestStruct struct {
	String    string                   `json:"string"`
	Number    float64                  `json:"number"`
	Bool      bool                     `json:"bool"`
	NullField interface{}              `json:"null_field"`
	Array     []int                    `json:"array"`
	Object    map[string]interface{}   `json:"object"`
	Nested    []map[string]interface{} `json:"nested"`
}

// 测试Unmarshal的基本功能
func TestUnmarshal(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected interface{}
	}{
		{
			name:     "null",
			input:    `null`,
			expected: nil,
		},
		{
			name:     "bool-true",
			input:    `true`,
			expected: true,
		},
		{
			name:     "bool-false",
			input:    `false`,
			expected: false,
		},
		{
			name:     "number-int",
			input:    `123`,
			expected: float64(123),
		},
		{
			name:     "number-float",
			input:    `-3.14`,
			expected: float64(-3.14),
		},
		{
			name:     "string",
			input:    `"hello"`,
			expected: "hello",
		},
		{
			name:     "array-empty",
			input:    `[]`,
			expected: []interface{}{},
		},
		{
			name:     "array-values",
			input:    `[1,2,3]`,
			expected: []interface{}{float64(1), float64(2), float64(3)},
		},
		{
			name:     "object-empty",
			input:    `{}`,
			expected: map[string]interface{}{},
		},
		{
			name:     "object-simple",
			input:    `{"name":"张三","age":30}`,
			expected: map[string]interface{}{"name": "张三", "age": float64(30)},
		},
		{
			name:     "object-complex",
			input:    `{"data":[1,2,{"key":"value"}]}`,
			expected: map[string]interface{}{"data": []interface{}{float64(1), float64(2), map[string]interface{}{"key": "value"}}},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var got interface{}
			err := Unmarshal([]byte(tc.input), &got)
			if err != nil {
				t.Errorf("Unmarshal(%q) 出错: %v", tc.input, err)
				return
			}

			// 深度比较结果
			if !compareValues(got, tc.expected) {
				t.Errorf("Unmarshal(%q) = %v, 期望 %v", tc.input, got, tc.expected)
			}
		})
	}
}

// 测试Unmarshal处理结构体
func TestUnmarshalStruct(t *testing.T) {
	jsonData := `{
		"string": "测试字符串",
		"number": 123.456,
		"bool": true,
		"null_field": null,
		"array": [1, 2, 3, 4, 5],
		"object": {
			"key1": "value1",
			"key2": 123
		},
		"nested": [
			{"id": 1, "name": "item1"},
			{"id": 2, "name": "item2"}
		]
	}`

	var result DecodeTestStruct
	err := Unmarshal([]byte(jsonData), &result)
	if err != nil {
		t.Fatalf("Unmarshal到结构体失败: %v", err)
	}

	// 验证结果
	if result.String != "测试字符串" {
		t.Errorf("String字段不匹配: 期望'测试字符串', 实际为'%s'", result.String)
	}
	if result.Number != 123.456 {
		t.Errorf("Number字段不匹配: 期望123.456, 实际为%f", result.Number)
	}
	if !result.Bool {
		t.Errorf("Bool字段不匹配: 期望true, 实际为%v", result.Bool)
	}
	if result.NullField != nil {
		t.Errorf("NullField字段不匹配: 期望nil, 实际为%v", result.NullField)
	}
	if len(result.Array) != 5 || result.Array[0] != 1 {
		t.Errorf("Array字段不匹配: %v", result.Array)
	}
	if len(result.Object) != 2 || result.Object["key1"] != "value1" {
		t.Errorf("Object字段不匹配: %v", result.Object)
	}
	if len(result.Nested) != 2 || result.Nested[0]["id"] != float64(1) {
		t.Errorf("Nested字段不匹配: %v", result.Nested)
	}
}

// 错误处理测试
func TestUnmarshalErrors(t *testing.T) {
	tests := []struct {
		name  string
		input string
	}{
		{
			name:  "invalid-json",
			input: `{"name": "missing quote}`,
		},
		{
			name:  "incomplete-object",
			input: `{"key": "value"`,
		},
		{
			name:  "extra-content",
			input: `{"key": "value"} extra`,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var result interface{}
			err := Unmarshal([]byte(tc.input), &result)
			if err == nil {
				t.Errorf("期望错误，但解析成功: %v", result)
			}
		})
	}
}

// 基准测试比较旧的解析方式和新的直接解析方式
func BenchmarkVsOldUnmarshal(b *testing.B) {
	// 测试数据
	jsonData := `{
		"string": "测试字符串",
		"number": 123.456,
		"bool": true,
		"null_field": null,
		"array": [1, 2, 3, 4, 5],
		"object": {
			"key1": "value1",
			"key2": 123,
			"key3": false,
			"nested": {
				"a": 1,
				"b": "字符串",
				"c": true,
				"d": [1, "2", false]
			}
		},
		"nested": [
			{"id": 1, "name": "item1", "value": 1},
			{"id": 2, "name": "item2", "value": 2},
			{"id": 3, "name": "item3", "value": 3}
		]
	}`

	// 禁用直接模式，测试旧解析方式
	oldConfig := defaultConfig

	// 测试旧解析方式
	b.Run("OldParser", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			var result DecodeTestStruct
			_ = UnmarshalWithConfig([]byte(jsonData), &result, oldConfig)
		}
	})

	// 测试新的直接解析方式
	b.Run("Parser", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			var result DecodeTestStruct
			_ = Unmarshal([]byte(jsonData), &result)
		}
	})

	// 作为比较参考的标准库解析
	b.Run("StdlibParser", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			var result DecodeTestStruct
			_ = json.Unmarshal([]byte(jsonData), &result)
		}
	})

	// 恢复默认配置
	defaultConfig = oldConfig
}

// 测试不同类型的JSON数据的性能
func BenchmarkUnmarshal(b *testing.B) {
	// 测试简单数据类型
	simpleJSON := `123`
	b.Run("Simple", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			var result interface{}
			_ = Unmarshal([]byte(simpleJSON), &result)
		}
	})

	// 测试小对象
	smallObjJSON := `{"name":"张三","age":30}`
	b.Run("SmallObject", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			var result map[string]interface{}
			_ = Unmarshal([]byte(smallObjJSON), &result)
		}
	})

	// 测试数组
	arrayJSON := `[1,2,3,4,5,6,7,8,9,10]`
	b.Run("Array", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			var result []interface{}
			_ = Unmarshal([]byte(arrayJSON), &result)
		}
	})

	// 测试嵌套对象
	nestedJSON := `{
		"name": "张三",
		"age": 30,
		"address": {
			"city": "北京",
			"street": "朝阳区",
			"zipcode": "100000"
		},
		"contacts": [
			{"type": "email", "value": "test@example.com"},
			{"type": "phone", "value": "123456789"}
		]
	}`
	b.Run("NestedObject", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			var result map[string]interface{}
			_ = Unmarshal([]byte(nestedJSON), &result)
		}
	})
}

// 辅助函数，用于深度比较值，处理浮点数等特殊情况
func compareValues(a, b interface{}) bool {
	if a == nil && b == nil {
		return true
	}
	if a == nil || b == nil {
		return false
	}

	switch va := a.(type) {
	case float64:
		if vb, ok := b.(float64); ok {
			return va == vb
		}
	case string:
		if vb, ok := b.(string); ok {
			return va == vb
		}
	case bool:
		if vb, ok := b.(bool); ok {
			return va == vb
		}
	case []interface{}:
		vb, ok := b.([]interface{})
		if !ok || len(va) != len(vb) {
			return false
		}
		for i := range va {
			if !compareValues(va[i], vb[i]) {
				return false
			}
		}
		return true
	case map[string]interface{}:
		vb, ok := b.(map[string]interface{})
		if !ok || len(va) != len(vb) {
			return false
		}
		for k, v1 := range va {
			v2, ok := vb[k]
			if !ok || !compareValues(v1, v2) {
				return false
			}
		}
		return true
	}

	return a == b
}
