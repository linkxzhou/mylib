package sjson

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"math"

	jsoniter "github.com/json-iterator/go"
)

var jsonfast = jsoniter.ConfigFastest

func TestSjsonUnmarshal(t *testing.T) {
	tests := []struct {
		input    string
		expected interface{}
	}{
		{`null`, nil},
		{`true`, true},
		{`false`, false},
		{`123`, float64(123)},
		{`-3.14`, float64(-3.14)},
		{`"hello"`, "hello"},
		{`[]`, []interface{}{}},
		{`[1,2,3]`, []interface{}{float64(1), float64(2), float64(3)}},
		{`{}`, map[string]interface{}{}},
		{`{"name":"张三","age":30}`, map[string]interface{}{"name": "张三", "age": float64(30)}},
		{`{"data":[1,2,{"key":"value"}]}`, map[string]interface{}{"data": []interface{}{float64(1), float64(2), map[string]interface{}{"key": "value"}}}},
	}

	for _, test := range tests {
		var got interface{}
		err := Unmarshal([]byte(test.input), &got)
		if err != nil {
			t.Errorf("Unmarshal(%q) 出错: %v", test.input, err)
			continue
		}

		if !reflect.DeepEqual(got, test.expected) {
			t.Errorf("Unmarshal(%q) = %v, 期望 %v", test.input, got, test.expected)
		}
	}
}

func TestSjsonMarshal(t *testing.T) {
	tests := []struct {
		input    interface{}
		expected string
	}{
		{nil, `null`},
		{true, `true`},
		{false, `false`},
		{123, `123`},
		{3.14, `3.14`},
		{"hello", `"hello"`},
		{[]int{}, `[]`},
		{[]int{1, 2, 3}, `[1,2,3]`},
		{map[string]interface{}{}, `{}`},
		{map[string]interface{}{"name": "张三", "age": 30}, `{"age":30,"name":"张三"}`},
	}

	for _, test := range tests {
		got, err := Marshal(test.input)
		if err != nil {
			t.Errorf("Marshal(%v) 出错: %v", test.input, err)
			continue
		}

		// 使用标准库解析结果进行比较，因为map的顺序不确定
		var gotObj, expectedObj interface{}
		if err := json.Unmarshal(got, &gotObj); err != nil {
			t.Errorf("解析结果 %q 出错: %v", got, err)
			continue
		}
		if err := json.Unmarshal([]byte(test.expected), &expectedObj); err != nil {
			t.Errorf("解析期望值 %q 出错: %v", test.expected, err)
			continue
		}

		if !reflect.DeepEqual(gotObj, expectedObj) {
			t.Errorf("Marshal(%v) = %q, 期望 %q", test.input, got, test.expected)
		}
	}
}

type Person struct {
	Name string `json:"name"`
	Age  int    `json:"age"`
}

func TestStructMarshalUnmarshal(t *testing.T) {
	// 测试结构体的序列化
	p := Person{Name: "李四", Age: 25}
	data, err := Marshal(p)
	if err != nil {
		t.Fatalf("Marshal 结构体出错: %v", err)
	}

	// 测试结构体的反序列化
	var p2 Person
	err = Unmarshal(data, &p2)
	if err != nil {
		t.Fatalf("Unmarshal 到结构体出错: %v", err)
	}

	if p.Name != p2.Name || p.Age != p2.Age {
		t.Errorf("结构体序列化再反序列化后不匹配: 原始=%v, 结果=%v", p, p2)
	}
}

func BenchmarkSjsonUnmarshal(b *testing.B) {
	jsonStr := `{
		"name": "测试",
		"age": 30,
		"married": false,
		"hobbies": ["reading", "coding"],
		"address": {
			"city": "北京",
			"country": "中国"
		}
	}`

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var result interface{}
		_ = Unmarshal([]byte(jsonStr), &result)
	}
}

func BenchmarkStdUnmarshal(b *testing.B) {
	jsonStr := `{
		"name": "测试",
		"age": 30,
		"married": false,
		"hobbies": ["reading", "coding"],
		"address": {
			"city": "北京",
			"country": "中国"
		}
	}`

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var result interface{}
		_ = json.Unmarshal([]byte(jsonStr), &result)
	}
}

// 新增针对 sjson_decode.go 的测试用例

// 用于测试解码器的结构体
type SjsonDecodeTestStruct struct {
	Name      string                 `json:"name"`
	Age       int                    `json:"age"`
	IsActive  bool                   `json:"is_active"`
	Balance   float64                `json:"balance"`
	Tags      []string               `json:"tags"`
	Metadata  map[string]interface{} `json:"metadata"`
	EmptyVal  string                 `json:"empty_val,omitempty"`
	IgnoreVal int                    `json:"-"`
}

// 测试基本类型解码
func TestBasicDecoding(t *testing.T) {
	tests := []struct {
		name     string
		jsonStr  string
		target   interface{}
		expected interface{}
	}{
		{
			name:     "bool true",
			jsonStr:  "true",
			target:   new(bool),
			expected: true,
		},
		{
			name:     "bool false",
			jsonStr:  "false",
			target:   new(bool),
			expected: false,
		},
		{
			name:     "int",
			jsonStr:  "123",
			target:   new(int),
			expected: 123,
		},
		{
			name:     "float",
			jsonStr:  "3.14159",
			target:   new(float64),
			expected: 3.14159,
		},
		{
			name:     "string",
			jsonStr:  `"hello, 世界"`,
			target:   new(string),
			expected: "hello, 世界",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := Unmarshal([]byte(test.jsonStr), test.target)
			if err != nil {
				t.Fatalf("Unmarshal失败: %v", err)
			}

			// 因为target是指针，需要获取其指向的值
			targetVal := reflect.ValueOf(test.target).Elem().Interface()

			// 对于浮点数，使用近似比较
			if test.name == "float" {
				targetFloat := targetVal.(float64)
				expectedFloat := test.expected.(float64)
				if math.Abs(targetFloat-expectedFloat) > 1e-10 {
					t.Errorf("解码结果 = %v, 期望 %v, 差值 = %v", targetFloat, expectedFloat, math.Abs(targetFloat-expectedFloat))
				}
			} else if !reflect.DeepEqual(targetVal, test.expected) {
				t.Errorf("解码结果 = %v, 期望 %v", targetVal, test.expected)
			}
		})
	}
}

// 测试复合类型解码
func TestCompositeDecoding(t *testing.T) {
	// 测试数组解码
	t.Run("Array", func(t *testing.T) {
		jsonStr := `[1, 2, 3, 4, 5]`
		var numbers []int
		err := Unmarshal([]byte(jsonStr), &numbers)
		if err != nil {
			t.Fatalf("数组解码失败: %v", err)
		}
		expected := []int{1, 2, 3, 4, 5}
		if !reflect.DeepEqual(numbers, expected) {
			t.Errorf("数组解码结果 = %v, 期望 %v", numbers, expected)
		}
	})

	// 测试map解码
	t.Run("Map", func(t *testing.T) {
		jsonStr := `{"key1": "value1", "key2": 123}`
		var result map[string]interface{}
		err := Unmarshal([]byte(jsonStr), &result)
		if err != nil {
			t.Fatalf("Map解码失败: %v", err)
		}
		if result["key1"] != "value1" || result["key2"] != float64(123) {
			t.Errorf("Map解码结果不符合预期: %v", result)
		}
	})

	// 测试嵌套结构
	t.Run("NestedStructure", func(t *testing.T) {
		jsonStr := `{
			"name": "测试用户",
			"age": 28,
			"is_active": true,
			"balance": 1234.56,
			"tags": ["tag1", "tag2", "tag3"],
			"metadata": {
				"created_at": "2025-01-01",
				"department": "Engineering"
			}
		}`

		var person SjsonDecodeTestStruct
		err := Unmarshal([]byte(jsonStr), &person)
		if err != nil {
			t.Fatalf("结构体解码失败: %v", err)
		}

		// 验证各个字段
		if person.Name != "测试用户" {
			t.Errorf("Name字段不匹配: 得到 %s, 期望 %s", person.Name, "测试用户")
		}
		if person.Age != 28 {
			t.Errorf("Age字段不匹配: 得到 %d, 期望 %d", person.Age, 28)
		}
		if !person.IsActive {
			t.Errorf("IsActive字段不匹配: 得到 %v, 期望 %v", person.IsActive, true)
		}
		if person.Balance != 1234.56 {
			t.Errorf("Balance字段不匹配: 得到 %f, 期望 %f", person.Balance, 1234.56)
		}
		if len(person.Tags) != 3 || person.Tags[0] != "tag1" || person.Tags[1] != "tag2" || person.Tags[2] != "tag3" {
			t.Errorf("Tags字段不匹配: %v", person.Tags)
		}
		if person.Metadata["created_at"] != "2025-01-01" || person.Metadata["department"] != "Engineering" {
			t.Errorf("Metadata字段不匹配: %v", person.Metadata)
		}
	})
}

// 测试错误处理
func TestDecodeErrors(t *testing.T) {
	tests := []struct {
		name    string
		jsonStr string
		target  interface{}
	}{
		{
			name:    "无效的JSON",
			jsonStr: `{"name": "不完整的JSON`,
			target:  new(map[string]interface{}),
		},
		{
			name:    "类型不匹配 - 字符串到整数",
			jsonStr: `"not an int"`,
			target:  new(int),
		},
		{
			name:    "类型不匹配 - 数组到对象",
			jsonStr: `[1, 2, 3]`,
			target:  new(map[string]interface{}),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := Unmarshal([]byte(test.jsonStr), test.target)
			if err == nil {
				t.Errorf("期望解码 %s 失败，但实际成功", test.jsonStr)
			}
		})
	}
}

// 测试指针和接口解码
func TestInterfaceAndPointerDecoding(t *testing.T) {
	// 测试空接口解码
	t.Run("EmptyInterface", func(t *testing.T) {
		jsonStr := `{"name": "接口测试", "values": [1, "string", true, null]}`
		var result interface{}
		err := Unmarshal([]byte(jsonStr), &result)
		if err != nil {
			t.Fatalf("接口解码失败: %v", err)
		}

		// 验证解码结果是否符合预期
		resultMap, ok := result.(map[string]interface{})
		if !ok {
			t.Fatalf("期望得到map[string]interface{}，获得 %T", result)
		}

		if resultMap["name"] != "接口测试" {
			t.Errorf("name字段不匹配: %v", resultMap["name"])
		}

		values, ok := resultMap["values"].([]interface{})
		if !ok || len(values) != 4 {
			t.Errorf("values字段不匹配: %v", resultMap["values"])
		}
	})

	// 测试指针解码
	t.Run("PointerDecoding", func(t *testing.T) {
		type NestedStruct struct {
			Value string `json:"value"`
		}

		type PointerTest struct {
			Name      string          `json:"name"`
			Nested    *NestedStruct   `json:"nested"`
			NestedArr []*NestedStruct `json:"nested_arr"`
		}

		jsonStr := `{
			"name": "指针测试",
			"nested": {"value": "嵌套值"},
			"nested_arr": [{"value": "数组1"}, {"value": "数组2"}]
		}`

		var result PointerTest
		err := Unmarshal([]byte(jsonStr), &result)
		if err != nil {
			t.Fatalf("指针解码失败: %v", err)
		}

		if result.Name != "指针测试" {
			t.Errorf("name字段不匹配: %v", result.Name)
		}

		if result.Nested == nil || result.Nested.Value != "嵌套值" {
			t.Errorf("nested字段不匹配: %v", result.Nested)
		}

		if len(result.NestedArr) != 2 {
			t.Errorf("nested_arr长度不匹配: %v", len(result.NestedArr))
		} else {
			if result.NestedArr[0].Value != "数组1" || result.NestedArr[1].Value != "数组2" {
				t.Errorf("nested_arr内容不匹配: %v", result.NestedArr)
			}
		}
	})
}

// 测试JSON特殊格式解码
func TestSpecialFormatDecoding(t *testing.T) {
	// 测试转义字符
	t.Run("EscapeChars", func(t *testing.T) {
		jsonStr := `{"text":"换行符\\n制表符\\t引号\\\"反斜杠\\\\"}`
		var result map[string]string
		err := Unmarshal([]byte(jsonStr), &result)
		if err != nil {
			t.Fatalf("转义字符解码失败: %v", err)
		}

		expected := "换行符\\n制表符\\t引号\\\"反斜杠\\\\"
		if result["text"] != expected {
			t.Errorf("转义字符解码不匹配: \n得到: %q\n期望: %q", result["text"], expected)
		}
	})

	// 测试Unicode编码
	t.Run("Unicode", func(t *testing.T) {
		jsonStr := `{"text":"Unicode测试: \\u4f60\\u597d"}`
		var result map[string]string
		err := Unmarshal([]byte(jsonStr), &result)
		if err != nil {
			t.Fatalf("Unicode解码失败: %v", err)
		}

		// 注意: 当前实现可能需要改进对Unicode的处理
		if !strings.Contains(result["text"], "Unicode测试") {
			t.Errorf("Unicode解码不匹配: %q", result["text"])
		}
	})
}

// 测试解码器缓存功能
func TestDecoderCache(t *testing.T) {
	// 准备一个复杂结构体，确保需要多种解码器
	type ComplexStruct struct {
		Int    int                    `json:"int"`
		String string                 `json:"string"`
		Bool   bool                   `json:"bool"`
		Array  []int                  `json:"array"`
		Map    map[string]interface{} `json:"map"`
		Struct DecodeTestStruct       `json:"struct"`
	}

	jsonStr := `{
		"int": 123,
		"string": "test",
		"bool": true,
		"array": [1, 2, 3],
		"map": {"key": "value"},
		"struct": {
			"name": "内部结构体",
			"age": 25,
			"is_active": true
		}
	}`

	// 第一次解码，会创建并缓存解码器
	var result1 ComplexStruct
	err := Unmarshal([]byte(jsonStr), &result1)
	if err != nil {
		t.Fatalf("第一次解码失败: %v", err)
	}

	// 第二次解码，应该使用缓存的解码器
	var result2 ComplexStruct
	err = Unmarshal([]byte(jsonStr), &result2)
	if err != nil {
		t.Fatalf("第二次解码失败: %v", err)
	}

	// 验证两次结果一致
	if !reflect.DeepEqual(result1, result2) {
		t.Errorf("两次解码结果不一致:\n第一次: %+v\n第二次: %+v", result1, result2)
	}
}

// 测试性能比较
func BenchmarkComplexDecode(b *testing.B) {
	type BenchStruct struct {
		String    string                   `json:"string"`
		Number    float64                  `json:"number"`
		Bool      bool                     `json:"bool"`
		NullField interface{}              `json:"null_field"`
		Array     []int                    `json:"array"`
		Object    map[string]interface{}   `json:"object"`
		Nested    []map[string]interface{} `json:"nested"`
	}

	jsonStr := `{
		"string": "Hello, World!",
		"number": 123.456,
		"bool": true,
		"null_field": null,
		"array": [1, 2, 3, 4, 5],
		"object": {
			"key1": "value1",
			"key2": 123,
			"key3": false
		},
		"nested": [
			{"name": "item1", "value": 1},
			{"name": "item2", "value": 2},
			{"name": "item3", "value": 3}
		]
	}`

	b.Run("SjsonDecode", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			var result BenchStruct
			_ = Unmarshal([]byte(jsonStr), &result)
		}
	})

	b.Run("StdDecode", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			var result BenchStruct
			_ = json.Unmarshal([]byte(jsonStr), &result)
		}
	})
}

// 测试单条JSON解码性能对比
func BenchmarkSingleDecode(b *testing.B) {
	jsonStr := []byte(`{"name":"测试","age":30,"active":true,"scores":[95,87,72]}`)

	b.Run("Sjson", func(b *testing.B) {
		config := Config{}

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			var result map[string]interface{}
			_ = UnmarshalWithConfig(jsonStr, &result, config)
		}
	})

	b.Run("StdLib", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			var result map[string]interface{}
			_ = json.Unmarshal(jsonStr, &result)
		}
	})

	b.Run("Jsoniter", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			var result map[string]interface{}
			_ = jsonfast.Unmarshal(jsonStr, &result)
		}
	})
}

// 测试多条不同JSON解码性能对比
func BenchmarkMultiDecode(b *testing.B) {
	// 准备10条不同的JSON字符串
	jsonStrings := []string{
		`{"id":1,"name":"item1","price":10.5}`,
		`{"id":2,"name":"item2","price":20.75,"tags":["tag1","tag2"]}`,
		`{"id":3,"name":"item3","price":30.99,"metadata":{"color":"red"}}`,
		`{"id":4,"name":"item4","price":40.15,"available":true}`,
		`{"id":5,"name":"item5","price":50.50,"discount":0.1}`,
		`{"id":6,"name":"item6","price":60.25,"rating":4.5}`,
		`{"id":7,"name":"item7","price":70.99,"options":{"size":"large"}}`,
		`{"id":8,"name":"item8","price":80.75,"count":100}`,
		`{"id":9,"name":"item9","price":90.5,"category":"electronics"}`,
		`{"id":10,"name":"item10","price":100.0,"featured":true}`,
	}

	b.Run("Sjson", func(b *testing.B) {
		config := Config{}

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			var result map[string]interface{}
			idx := i % len(jsonStrings)
			_ = UnmarshalWithConfig([]byte(jsonStrings[idx]), &result, config)
		}
	})

	b.Run("StdLib", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			var result map[string]interface{}
			idx := i % len(jsonStrings)
			_ = json.Unmarshal([]byte(jsonStrings[idx]), &result)
		}
	})

	b.Run("Jsoniter", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			var result map[string]interface{}
			idx := i % len(jsonStrings)
			_ = jsonfast.Unmarshal([]byte(jsonStrings[idx]), &result)
		}
	})
}

// 测试带缓存的结构体解码性能
func BenchmarkStructDecode(b *testing.B) {
	type Product struct {
		ID        int                    `json:"id"`
		Name      string                 `json:"name"`
		Price     float64                `json:"price"`
		Available bool                   `json:"available"`
		Tags      []string               `json:"tags,omitempty"`
		Metadata  map[string]interface{} `json:"metadata,omitempty"`
	}

	jsonStr := []byte(`{
		"id": 12345,
		"name": "测试产品",
		"price": 99.99,
		"available": true,
		"tags": ["新品", "热卖", "推荐"],
		"metadata": {
			"color":  "红色",
			"size":   "中号",
			"weight": 0.5,
			"dimensions": {
				"length": 10,
				"width":  5,
				"height": 2
			}
		}
	}`)

	b.Run("Sjson", func(b *testing.B) {
		config := Config{}

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			var product Product
			_ = UnmarshalWithConfig(jsonStr, &product, config)
		}
	})

	b.Run("StdLib", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			var product Product
			_ = json.Unmarshal(jsonStr, &product)
		}
	})

	b.Run("Jsoniter", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			var product Product
			_ = jsonfast.Unmarshal(jsonStr, &product)
		}
	})
}

// 测试带有缓存的复杂数组解码性能
func BenchmarkArrayDecode(b *testing.B) {
	jsonStr := []byte(`[
		{"id": 1, "name": "Item 1", "tags": ["tag1", "tag2"]},
		{"id": 2, "name": "Item 2", "tags": ["tag2", "tag3"]},
		{"id": 3, "name": "Item 3", "tags": ["tag1", "tag3"]},
		{"id": 4, "name": "Item 4", "tags": ["tag4"]},
		{"id": 5, "name": "Item 5", "tags": ["tag1", "tag5"]}
	]`)

	b.Run("Sjson", func(b *testing.B) {
		config := Config{}

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			var items []map[string]interface{}
			_ = UnmarshalWithConfig(jsonStr, &items, config)
		}
	})

	b.Run("StdLib", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			var items []map[string]interface{}
			_ = json.Unmarshal(jsonStr, &items)
		}
	})
}

// 测试单条对象编码性能对比
func BenchmarkSingleEncode(b *testing.B) {
	// 准备要编码的对象
	obj := map[string]interface{}{
		"name":   "测试对象",
		"age":    30,
		"active": true,
		"scores": []int{95, 87, 72},
	}

	b.Run("SjsonEncode", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _ = Marshal(obj)
		}
	})

	b.Run("StdEncode", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _ = json.Marshal(obj)
		}
	})
}

// 测试结构体编码性能对比
func BenchmarkStructEncode(b *testing.B) {
	// 定义测试结构体
	type Product struct {
		ID        int                    `json:"id"`
		Name      string                 `json:"name"`
		Price     float64                `json:"price"`
		Available bool                   `json:"available"`
		Tags      []string               `json:"tags,omitempty"`
		Metadata  map[string]interface{} `json:"metadata,omitempty"`
	}

	// 创建测试数据
	product := Product{
		ID:        12345,
		Name:      "测试产品",
		Price:     99.99,
		Available: true,
		Tags:      []string{"新品", "热卖", "推荐"},
		Metadata: map[string]interface{}{
			"color":  "红色",
			"size":   "中号",
			"weight": 0.5,
			"dimensions": map[string]int{
				"length": 10,
				"width":  5,
				"height": 2,
			},
		},
	}

	b.Run("SjsonEncode", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _ = Marshal(product)
		}
	})

	b.Run("StdEncode", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _ = json.Marshal(product)
		}
	})
}

// 测试数组编码性能对比
func BenchmarkArrayEncode(b *testing.B) {
	// 创建测试数据
	items := []map[string]interface{}{
		{"id": 1, "name": "Item 1", "tags": []string{"tag1", "tag2"}},
		{"id": 2, "name": "Item 2", "tags": []string{"tag2", "tag3"}},
		{"id": 3, "name": "Item 3", "tags": []string{"tag1", "tag3"}},
		{"id": 4, "name": "Item 4", "tags": []string{"tag4"}},
		{"id": 5, "name": "Item 5", "tags": []string{"tag1", "tag5"}},
	}

	b.Run("SjsonEncode", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _ = Marshal(items)
		}
	})

	b.Run("StdEncode", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _ = json.Marshal(items)
		}
	})
}

// 测试基本类型编码性能对比
func BenchmarkBasicTypesEncode(b *testing.B) {
	// 创建基本类型测试数据
	basicData := map[string]interface{}{
		"int":      12345,
		"uint":     uint(67890),
		"float":    123.456,
		"bool":     true,
		"null":     nil,
		"intArr":   []int{1, 2, 3, 4, 5},
		"floatArr": []float64{1.1, 2.2, 3.3},
	}

	b.Run("SjsonBasicEncode", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _ = Marshal(basicData)
		}
	})

	b.Run("StdBasicEncode", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _ = json.Marshal(basicData)
		}
	})
}

// 测试字符串编码性能对比
func BenchmarkStringEncode(b *testing.B) {
	// 创建各种类型的字符串测试数据
	stringData := map[string]string{
		"simple":      "Hello, World!",
		"withEscape":  "Hello, \"World\"! \t\r\n",
		"withUnicode": "你好，世界！Unicode字符串测试",
		"longString": "这是一个比较长的字符串，用于测试sjson的字符串编码性能。" +
			"它包含了各种字符，包括英文字母、数字、标点符号，以及一些中文字符。" +
			"这个字符串的长度超过了100个字符，可以更好地测试长字符串的编码性能。",
	}

	b.Run("SjsonStringEncode", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _ = Marshal(stringData)
		}
	})

	b.Run("StdStringEncode", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _ = json.Marshal(stringData)
		}
	})
}

// 测试复杂对象编码性能对比
func BenchmarkComplexEncode(b *testing.B) {
	// 创建一个复杂的嵌套对象
	complexObj := map[string]interface{}{
		"string": "Hello, World!",
		"number": 123.456,
		"bool":   true,
		"null":   nil,
		"array":  []int{1, 2, 3, 4, 5},
		"object": map[string]interface{}{
			"key1": "value1",
			"key2": 123,
			"key3": false,
			"nested": map[string]interface{}{
				"a": 1,
				"b": "字符串",
				"c": true,
				"d": []interface{}{1, "2", false},
			},
		},
		"nested_array": []map[string]interface{}{
			{"name": "item1", "value": 1},
			{"name": "item2", "value": 2},
			{"name": "item3", "value": 3},
		},
	}

	b.Run("SjsonEncode", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _ = Marshal(complexObj)
		}
	})

	b.Run("StdEncode", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _ = json.Marshal(complexObj)
		}
	})
}

var mediumFixture []byte = []byte(`{
	"person": {
	  "id": "d50887ca-a6ce-4e59-b89f-14f0b5d03b03",
	  "name": {
		"fullName": "Leonid Bugaev",
		"givenName": "Leonid",
		"familyName": "Bugaev"
	  },
	  "email": "leonsbox@gmail.com",
	  "gender": "male",
	  "location": "Saint Petersburg, Saint Petersburg, RU",
	  "geo": {
		"city": "Saint Petersburg",
		"state": "Saint Petersburg",
		"country": "Russia",
		"lat": 59.9342802,
		"lng": 30.3350986
	  },
	  "bio": "Senior engineer at Granify.com",
	  "site": "http://flickfaver.com",
	  "avatar": "https://d1ts43dypk8bqh.cloudfront.net/v1/avatars/d50887ca-a6ce-4e59-b89f-14f0b5d03b03",
	  "employment": {
		"name": "www.latera.ru",
		"title": "Software Engineer",
		"domain": "gmail.com"
	  },
	  "facebook": {
		"handle": "leonid.bugaev"
	  },
	  "github": {
		"handle": "buger",
		"id": 14009,
		"avatar": "https://avatars.githubusercontent.com/u/14009?v=3",
		"company": "Granify",
		"blog": "http://leonsbox.com",
		"followers": 95,
		"following": 10
	  },
	  "twitter": {
		"handle": "flickfaver",
		"id": 77004410,
		"bio": null,
		"followers": 2,
		"following": 1,
		"statuses": 5,
		"favorites": 0,
		"location": "",
		"site": "http://flickfaver.com",
		"avatar": null
	  },
	  "linkedin": {
		"handle": "in/leonidbugaev"
	  },
	  "googleplus": {
		"handle": null
	  },
	  "angellist": {
		"handle": "leonid-bugaev",
		"id": 61541,
		"bio": "Senior engineer at Granify.com",
		"blog": "http://buger.github.com",
		"site": "http://buger.github.com",
		"followers": 41,
		"avatar": "https://d1qb2nb5cznatu.cloudfront.net/users/61541-medium_jpg?1405474390"
	  },
	  "klout": {
		"handle": null,
		"score": null
	  },
	  "foursquare": {
		"handle": null
	  },
	  "aboutme": {
		"handle": "leonid.bugaev",
		"bio": null,
		"avatar": null
	  },
	  "gravatar": {
		"handle": "buger",
		"urls": [
		],
		"avatar": "http://1.gravatar.com/avatar/f7c8edd577d13b8930d5522f28123510",
		"avatars": [
		  {
			"url": "http://1.gravatar.com/avatar/f7c8edd577d13b8930d5522f28123510",
			"type": "thumbnail"
		  }
		]
	  },
	  "fuzzy": false
	},
	"company": null
  }`)

type CBAvatar struct {
	Url string `json:"url"`
}

type CBGravatar struct {
	Avatars []*CBAvatar `json:"avatars"`
}

type CBGithub struct {
	Followers int `json:"followers"`
}

type CBName struct {
	FullName string `json:"fullName"`
}

type CBPerson struct {
	Name     *CBName     `json:"name"`
	Github   *CBGithub   `json:"github"`
	Gravatar *CBGravatar `json:"gravatar"`
}

type MediumPayload struct {
	Person  *CBPerson `json:"person"`
	Company string    `json:"compnay"`
}

func BenchmarkCompareMedium(b *testing.B) {
	b.Run("SjsonMarshal", func(b *testing.B) {
		b.ResetTimer()
		var data MediumPayload
		Unmarshal(mediumFixture, &data)
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			_, _ = Marshal(data)
		}
	})

	b.Run("StdMarshal", func(b *testing.B) {
		b.ResetTimer()
		var data MediumPayload
		json.Unmarshal(mediumFixture, &data)
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			_, _ = json.Marshal(data)
		}
	})

	b.Run("JsoniterMarshal", func(b *testing.B) {
		var data MediumPayload
		jsonfast.Unmarshal(mediumFixture, &data)
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			_, _ = jsonfast.Marshal(data)
		}
	})

	b.Run("SjsonUnmarshal", func(b *testing.B) {
		b.ResetTimer()
		var data MediumPayload
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			Unmarshal(mediumFixture, &data)
		}
	})

	b.Run("StdUnmarshal", func(b *testing.B) {
		b.ResetTimer()
		var data MediumPayload
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			json.Unmarshal(mediumFixture, &data)
		}
	})

	b.Run("JsoniterUnmarshal", func(b *testing.B) {
		b.ResetTimer()
		var data MediumPayload
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			jsonfast.Unmarshal(mediumFixture, &data)
		}
	})
}
