package importer

import (
	"go/ast"
	"reflect"
)

// BasicKind describes the kind of basic type.
type BasicKind int

const (
	// Unknown 未知
	Unknown BasicKind = iota
	// Var 变量
	Var
	// Const 常量
	Const
	// TypeName 类型
	TypeName
	// Function 函数
	Function
	// BuiltinFunction 内置函数
	BuiltinFunction
)

// ExternalPackage 外部引入包
type ExternalPackage struct {
	Path    string
	Name    string
	Objects []*ExternalObject
}

// ExternalObject 外部引入对象
type ExternalObject struct {
	Name  string
	Kind  BasicKind
	Value reflect.Value
	Type  reflect.Type
	Doc   string
}

var packages = make(map[string]*ExternalPackage)

// 按名称索引的包描述，用于自动添加import语句
var packagesByName = make(map[string]*ast.ImportSpec)

// GetPackageByName 根据名称获取包信息
func GetPackageByName(name string) *ast.ImportSpec {
	return packagesByName[name]
}

// GetAllPackages 获取所有包
func GetAllPackages() map[string]*ExternalPackage {
	return packages
}
