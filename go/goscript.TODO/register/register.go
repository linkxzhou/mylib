package register

import (
	"reflect"

	"git.woa.com/vasd_masc_ba/YitihuaOteam/base/gofun/internal/importer"
)

// AddPackage 注册package
// 使用时需要保证先添加完所有package，再进行代码解析，
func AddPackage(path string, name string, objects ...*importer.ExternalObject) {
	importer.RegisterPackage(path, name, objects...)
}

// NewFunction 新增函数
func NewFunction(name string, value interface{}, doc string) *importer.ExternalObject {
	return &importer.ExternalObject{
		Name:  name,
		Kind:  importer.Function,
		Value: reflect.ValueOf(value),
		Type:  reflect.TypeOf(value),
		Doc:   doc,
	}
}

// NewVar 新增变量
func NewVar(name string, valueAddr interface{}, typ reflect.Type, doc string) *importer.ExternalObject {
	return &importer.ExternalObject{
		Name:  name,
		Kind:  importer.Var,
		Value: reflect.ValueOf(valueAddr),
		Type:  typ,
		Doc:   doc,
	}
}

// NewConst 新增常量
func NewConst(name string, value interface{}, doc string) *importer.ExternalObject {
	return &importer.ExternalObject{
		Name:  name,
		Kind:  importer.Const,
		Value: reflect.ValueOf(value),
		Type:  reflect.TypeOf(value),
		Doc:   doc,
	}
}

// NewType 新增类型
func NewType(name string, typ reflect.Type, doc string) *importer.ExternalObject {
	return &importer.ExternalObject{
		Name: name,
		Kind: importer.TypeName,
		Type: typ,
		Doc:  doc,
	}
}
