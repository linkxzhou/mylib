package sjson

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"testing"
)

// 测试用复杂结构体
type EncodeTestStruct struct {
	Name      string                 `json:"name"`
	Age       int                    `json:"age"`
	IsActive  bool                   `json:"is_active"`
	Balance   float64                `json:"balance"`
	Tags      []string               `json:"tags"`
	Metadata  map[string]interface{} `json:"metadata"`
	EmptyVal  string                 `json:"empty_val,omitempty"`
	IgnoreVal int                    `json:"-"`
}

// 嵌套结构体测试
type NestedStruct struct {
	ID       int                `json:"id"`
	Parent   *EncodeTestStruct  `json:"parent"`
	Children []EncodeTestStruct `json:"children"`
	Extra    map[string][]int   `json:"extra"`
}

// 测试Marshal与标准库json.Marshal结果一致性
func TestMarshalMatchesMarshal(t *testing.T) {
	tests := []interface{}{
		nil,
		true,
		false,
		123,
		-3.14,
		"测试字符串",
		[]int{1, 2, 3},
		[]interface{}{1, "abc", true, nil},
		map[string]int{"one": 1, "two": 2},
		map[string]interface{}{"name": "张三", "age": 30, "data": []interface{}{1, 2}},
		EncodeTestStruct{
			Name:     "李四",
			Age:      25,
			IsActive: true,
			Balance:  123.45,
			Tags:     []string{"tag1", "tag2"},
			Metadata: map[string]interface{}{"key1": "value1", "key2": 123},
		},
		NestedStruct{
			ID: 1,
			Parent: &EncodeTestStruct{
				Name: "父对象",
				Age:  50,
			},
			Children: []EncodeTestStruct{
				{Name: "子对象1", Age: 10},
				{Name: "子对象2", Age: 15},
			},
			Extra: map[string][]int{
				"scores": {95, 98, 100},
			},
		},
	}

	for _, test := range tests {
		// 使用标准库json.Marshal方法
		stdJSON, err := json.Marshal(test)
		if err != nil {
			t.Errorf("标准库json.Marshal(%+v) 失败: %v", test, err)
			continue
		}

		// 使用自己实现的Marshal方法
		ourJSON, err := Marshal(test)
		if err != nil {
			t.Errorf("我们的Marshal(%+v) 失败: %v", test, err)
			continue
		}

		// 只进行解码后的对象比较，避免格式差异导致字符串不同
		var stdObj, ourObj interface{}
		if err := json.Unmarshal(stdJSON, &stdObj); err != nil {
			t.Errorf("解析标准库JSON失败: %v, JSON: %s", err, stdJSON)
			continue
		}

		if err := json.Unmarshal(ourJSON, &ourObj); err != nil {
			t.Errorf("解析我们的Marshal JSON失败: %v, JSON: %s", err, ourJSON)
			continue
		}

		if !reflect.DeepEqual(stdObj, ourObj) {
			t.Errorf("Marshal结果与标准库不兼容\nMarshal: %s\nStdJSON: %s", ourJSON, stdJSON)
		}
	}
}

// 测试MarshalString正确性
func TestMarshalString(t *testing.T) {
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
		{map[string]int{}, `{}`},
		{map[string]interface{}{"name": "张三"}, `{"name":"张三"}`},
	}

	for _, test := range tests {
		// 直接获取字符串
		result, err := MarshalString(test.input)
		if err != nil {
			t.Errorf("MarshalString(%v) 失败: %v", test.input, err)
			continue
		}

		if result != test.expected {
			t.Errorf("MarshalString(%v) = %s, 期望 %s", test.input, result, test.expected)
		}
	}
}

// 测试特殊字符处理
func TestMarshalSpecialChars(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{`"quoted"`, `"\"quoted\""`},
		{"\n\t\r", `"\n\t\r"`},
		{"中文和特殊字符：<>&", `"中文和特殊字符：<>&"`},
		{"\u0000\u001f", `"\u0000\u001f"`},
	}

	for _, test := range tests {
		result, err := MarshalString(test.input)
		if err != nil {
			t.Errorf("MarshalString(%q) 失败: %v", test.input, err)
			continue
		}

		if result != test.expected {
			t.Errorf("MarshalString(%q) = %s, 期望 %s", test.input, result, test.expected)
		}
	}
}

// 测试omitempty标签
func TestMarshalOmitEmpty(t *testing.T) {
	// 测试空值
	empty := EncodeTestStruct{
		Name: "测试omitempty",
		// EmptyVal保持为空
	}

	result, err := MarshalString(empty)
	if err != nil {
		t.Errorf("MarshalString(empty) 失败: %v", err)
	}

	// 检查结果中不应该包含empty_val字段
	if result != `{"name":"测试omitempty","age":0,"is_active":false,"balance":0,"tags":null,"metadata":null}` {
		t.Errorf("MarshalString未正确处理omitempty标签: %s", result)
	}

	// 测试非空值
	notEmpty := EncodeTestStruct{
		Name:     "测试omitempty",
		EmptyVal: "有值",
	}

	result, err = MarshalString(notEmpty)
	if err != nil {
		t.Errorf("MarshalString(notEmpty) 失败: %v", err)
	}

	// 检查结果中应该包含empty_val字段
	if !strings.Contains(result, `"empty_val":"有值"`) {
		t.Errorf("MarshalString未包含非空的omitempty字段: %s", result)
	}
}

// 测试排序键功能
func TestMarshalSortedKeys(t *testing.T) {
	// 创建一个带有多个键的map
	testMap := map[string]int{
		"z": 26,
		"a": 1,
		"m": 13,
	}

	// 设置排序
	origSortSetting := defaultConfig.SortMapKeys
	defaultConfig.SortMapKeys = true
	defer func() {
		defaultConfig.SortMapKeys = origSortSetting
	}()

	// 序列化
	result, err := MarshalString(testMap)
	if err != nil {
		t.Errorf("MarshalString(sortedMap) 失败: %v", err)
	}

	// 排序后应该是 {"a":1,"m":13,"z":26}
	expected := `{"a":1,"m":13,"z":26}`
	if result != expected {
		t.Errorf("MarshalString未正确排序键: 得到 %s, 期望 %s", result, expected)
	}
}

// --------- 性能基准测试 ---------

// 基本类型编码基准测试
func BenchmarkBasicTypes(b *testing.B) {
	// 数字
	b.Run("Marshal-Number", func(b *testing.B) {
		number := 12345.6789
		for i := 0; i < b.N; i++ {
			_, _ = Marshal(number)
		}
	})

	b.Run("StdMarshal-Number", func(b *testing.B) {
		number := 12345.6789
		for i := 0; i < b.N; i++ {
			_, _ = json.Marshal(number)
		}
	})

	// 字符串
	b.Run("Marshal-String", func(b *testing.B) {
		s := "这是一个性能测试用的字符串，需要包含一些中文和特殊字符!@#$%^&*()"
		for i := 0; i < b.N; i++ {
			_, _ = Marshal(s)
		}
	})

	b.Run("StdMarshal-String", func(b *testing.B) {
		s := "这是一个性能测试用的字符串，需要包含一些中文和特殊字符!@#$%^&*()"
		for i := 0; i < b.N; i++ {
			_, _ = json.Marshal(s)
		}
	})
}

// 复杂结构编码基准测试
func BenchmarkComplexTypes(b *testing.B) {
	// 复杂结构体
	complexObj := NestedStruct{
		ID: 1,
		Parent: &EncodeTestStruct{
			Name:     "父对象",
			Age:      50,
			IsActive: true,
			Balance:  999.99,
			Tags:     []string{"parent", "important"},
			Metadata: map[string]interface{}{
				"level": "high",
				"score": 98,
				"attrs": []interface{}{true, 123, "属性"},
			},
		},
		Children: []EncodeTestStruct{
			{
				Name:     "子对象1",
				Age:      10,
				IsActive: true,
				Tags:     []string{"child", "active"},
			},
			{
				Name:     "子对象2",
				Age:      15,
				IsActive: false,
				Tags:     []string{"child", "inactive"},
			},
		},
		Extra: map[string][]int{
			"scores":  {90, 95, 100},
			"ranking": {1, 3, 5},
		},
	}

	b.Run("Marshal-Complex", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, _ = Marshal(complexObj)
		}
	})

	b.Run("StdMarshal-Complex", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, _ = json.Marshal(complexObj)
		}
	})
}

// 大型数组编码基准测试
func BenchmarkLargeArray(b *testing.B) {
	// 生成大型数组
	largeArray := make([]int, 1000)
	for i := 0; i < 1000; i++ {
		largeArray[i] = i
	}

	b.Run("Marshal-LargeArray", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, _ = Marshal(largeArray)
		}
	})

	b.Run("StdMarshal-LargeArray", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, _ = json.Marshal(largeArray)
		}
	})
}

// 大型Map编码基准测试
func BenchmarkLargeMap(b *testing.B) {
	// 生成大型Map
	largeMap := make(map[string]int, 1000)
	for i := 0; i < 1000; i++ {
		largeMap[fmt.Sprintf("key%d", i)] = i
	}

	b.Run("Marshal-LargeMap", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, _ = Marshal(largeMap)
		}
	})

	b.Run("StdMarshal-LargeMap", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, _ = json.Marshal(largeMap)
		}
	})
}

// 字符串直接编码性能对比
func BenchmarkMarshalString(b *testing.B) {
	complexObj := NestedStruct{
		ID: 1,
		Parent: &EncodeTestStruct{
			Name:     "父对象",
			Age:      50,
			IsActive: true,
		},
		Children: []EncodeTestStruct{
			{Name: "子对象1", Age: 10},
			{Name: "子对象2", Age: 15},
		},
	}

	b.Run("MarshalString", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, _ = MarshalString(complexObj)
		}
	})

	b.Run("Marshal+ToString", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			data, _ := Marshal(complexObj)
			_ = string(data)
		}
	})

	b.Run("StdMarshal+ToString", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			data, _ := json.Marshal(complexObj)
			_ = string(data)
		}
	})
}
