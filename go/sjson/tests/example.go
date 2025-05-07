package main

import (
	"fmt"
	"io/ioutil"
	"os"

	"github.com/mylib/go/sjson"
)

func parse_json_file(file_path string) {
	b, file_err := ioutil.ReadFile(file_path)
	if file_err != nil {
		os.Exit(-1)
	}

	var f interface{}
	err := sjson.Unmarshal(b, &f)

	if err != nil {
		os.Exit(1)
	}

	os.Exit(0)
}

func parse_json_string() {
	// 示例1: 解析简单的JSON字符串
	fmt.Println("示例1: 解析简单的JSON字符串")
	jsonStr := `{"name":"张三","age":30,"skills":["Go","Python"]}`

	var data interface{}
	err := sjson.Unmarshal([]byte(jsonStr), &data)
	if err != nil {
		fmt.Println("解析错误:", err)
		os.Exit(1)
	}
	fmt.Printf("解析结果: %+v\n\n", data)

	// 示例2: 解析JSON到结构体
	fmt.Println("示例2: 解析JSON到结构体")
	type Person struct {
		Name   string   `json:"name"`
		Age    int      `json:"age"`
		Skills []string `json:"skills"`
	}

	var person Person
	err = sjson.Unmarshal([]byte(jsonStr), &person)
	if err != nil {
		fmt.Println("解析错误:", err)
		os.Exit(1)
	}
	fmt.Printf("姓名: %s, 年龄: %d, 技能: %v\n\n", person.Name, person.Age, person.Skills)

	// 示例3: 生成JSON
	fmt.Println("示例3: 生成JSON")
	newPerson := Person{
		Name:   "李四",
		Age:    25,
		Skills: []string{"Java", "C++"},
	}

	data, err = sjson.Marshal(newPerson)
	if err != nil {
		fmt.Println("编码错误:", err)
		os.Exit(1)
	}
	fmt.Printf("生成的JSON: %v\n\n", data)

	// 手动构建JSON
	fmt.Println("\n示例4: 手动构建JSON")
	manualObj := sjson.Object{
		"status": sjson.String("ok"),
		"code":   sjson.Number(200),
		"user": sjson.Object{
			"id":   sjson.Number(1001),
			"name": sjson.String("王五"),
		},
		"tags": sjson.Array{
			sjson.String("标签1"),
			sjson.String("标签2"),
		},
	}

	fmt.Printf("手动构建的JSON: %s\n", manualObj.String())
}

func main() {
	if len(os.Args) < 2 {
		parse_json_string()
	} else {
		parse_json_file(os.Args[1])
	}
}
