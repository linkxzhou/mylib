# sjson

## 功能

sjson 是一个参考 json-iterator 实现的 Go 语言 JSON 解析库，提供了高效的 JSON 编码和解码功能。

## 特性

- 简单易用的 API，与标准库 `encoding/json` 接口兼容
- 支持基本的 JSON 数据类型：null、布尔值、数字、字符串、数组和对象
- 支持结构体与 JSON 的相互转换，支持 `json` 标签
- 提供流式解析功能，可从字符串或 Reader 中解析 JSON

## 安装

```bash
go get github.com/mylib/go/sjson
```

## 使用示例

### 解析 JSON 字符串

```go
package main

import (
	"fmt"
	"github.com/mylib/go/sjson"
)

func main() {
	// 解析 JSON 到 interface{}
	var data interface{}
	jsonStr := `{"name":"张三","age":30,"skills":["Go","Python"]}`
	err := sjson.Unmarshal([]byte(jsonStr), &data)
	if err != nil {
		fmt.Println("解析错误:", err)
		return
	}
	fmt.Printf("%+v\n", data)

	// 解析 JSON 到结构体
	type Person struct {
		Name   string   `json:"name"`
		Age    int      `json:"age"`
		Skills []string `json:"skills"`
	}

	var person Person
	err = sjson.Unmarshal([]byte(jsonStr), &person)
	if err != nil {
		fmt.Println("解析错误:", err)
		return
	}
	fmt.Printf("%+v\n", person)
}
```

### 生成 JSON 字符串

```go
package main

import (
	"fmt"
	"github.com/mylib/go/sjson"
)

func main() {
	// 从结构体生成 JSON
	person := struct {
		Name   string   `json:"name"`
		Age    int      `json:"age"`
		Skills []string `json:"skills"`
	}{
		Name:   "李四",
		Age:    25,
		Skills: []string{"Java", "C++"},
	}

	data, err := sjson.Marshal(person)
	if err != nil {
		fmt.Println("编码错误:", err)
		return
	}
	fmt.Println(string(data))
}
```

### 使用底层 API

```go
package main

import (
	"fmt"
	"github.com/mylib/go/sjson"
)

func main() {
	// 直接解析 JSON 字符串
	jsonStr := `{"success":true,"data":{"items":[1,2,3]}}`
	value, err := sjson.ParseString(jsonStr)
	if err != nil {
		fmt.Println("解析错误:", err)
		return
	}

	// 访问解析后的值
	obj := value.(sjson.Object)
	success := obj["success"].(sjson.Bool)
	data := obj["data"].(sjson.Object)
	items := data["items"].(sjson.Array)

	fmt.Println("success:", bool(success))
	fmt.Println("items:", items)
}
```

## API 文档

### 主要函数

- `Unmarshal(data []byte, v interface{}) error` - 将 JSON 字节切片解析为 Go 对象
- `Marshal(v interface{}) ([]byte, error)` - 将 Go 对象编码为 JSON 字节切片
- `ParseString(s string) (Value, error)` - 解析 JSON 字符串并返回对应的值
- `ParseReader(r io.Reader) (Value, error)` - 从 io.Reader 解析 JSON 并返回对应的值

### 数据类型

- `Value` - 表示一个 JSON 值的接口
- `Null` - 表示 JSON 中的 null 值
- `Bool` - 表示 JSON 中的布尔值
- `Number` - 表示 JSON 中的数值
- `String` - 表示 JSON 中的字符串
- `Array` - 表示 JSON 中的数组
- `Object` - 表示 JSON 中的对象

## 性能

sjson 库的性能目标是接近标准库 `encoding/json`，同时提供更简洁的 API 和更好的可扩展性。

