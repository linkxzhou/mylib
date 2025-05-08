package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"

	jsoniter "github.com/json-iterator/go"
	"github.com/mylib/go/sjson"
)

func parse_sjson_file(file_path string) {
	b, file_err := ioutil.ReadFile(file_path)
	if file_err != nil {
		fmt.Println("读取文件错误:", file_err)
		os.Exit(-1)
	}

	// 使用自定义sjson库解析
	var fSjson interface{}
	errSjson := sjson.Unmarshal(b, &fSjson)
	if errSjson != nil {
		os.Exit(1)
	}

	os.Exit(0)
}

func parse_stdjson_file(file_path string) {
	b, file_err := ioutil.ReadFile(file_path)
	if file_err != nil {
		fmt.Println("读取文件错误:", file_err)
		os.Exit(-1)
	}

	// 使用标准库json解析
	var fStd interface{}
	errStd := json.Unmarshal(b, &fStd)
	if errStd != nil {
		os.Exit(1)
	}

	os.Exit(0)
}

func parse_fastjson_file(file_path string) {
	b, file_err := ioutil.ReadFile(file_path)
	if file_err != nil {
		fmt.Println("读取文件错误:", file_err)
		os.Exit(-1)
	}

	// 使用json-iterator库解析
	var fJsoniter interface{}
	errJsoniter := jsoniter.Unmarshal(b, &fJsoniter)
	if errJsoniter != nil {
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
}

func main() {
	if len(os.Args) < 2 {
		parse_json_string()
		return
	}

	// 解析命令行参数
	arg := os.Args[1]
	if arg == "-a" {
		// 如果参数是-a，使用sjson解析
		if len(os.Args) < 3 {
			fmt.Println("请提供要解析的JSON文件路径")
			os.Exit(1)
		}
		parse_sjson_file(os.Args[2])
	} else if arg == "-b" {
		// 如果参数是-b，使用标准库json解析
		if len(os.Args) < 3 {
			fmt.Println("请提供要解析的JSON文件路径")
			os.Exit(1)
		}
		parse_stdjson_file(os.Args[2])
	} else if arg == "-c" {
		// 如果参数是-c，使用json-iterator解析
		if len(os.Args) < 3 {
			fmt.Println("请提供要解析的JSON文件路径")
			os.Exit(1)
		}
		parse_fastjson_file(os.Args[2])
	} else {
		// 默认使用sjson解析
		parse_sjson_file(arg)
	}
}
