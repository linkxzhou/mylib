package value

import (
	_ "fmt"
	"reflect"
	"strings"

	"git.woa.com/vasd_masc_ba/YitihuaOteam/base/gofun/internal/importer"
	"golang.org/x/tools/go/ssa"
	"golang.org/x/tools/go/ssa/ssautil"
)

type Value interface {
	Elem() Value
	Interface() interface{}
	String() string
	Int() int64
	Uint() uint64
	Float() float64
	Index(i int) Value
	MapIndex(v Value) Value
	Set(Value)
	Len() int
	Cap() int
	Type() reflect.Type
	IsValid() bool
	IsNil() bool
	Bool() bool
	Field(i int) Value
	Next() Value
	Kind() reflect.Kind

	RValue() reflect.Value
}

// ExternalValue 外部引入的变量
type ExternalValue struct {
	ssa.Value
	Object *importer.ExternalObject
}

// Store 修改外部变量的值
func (p *ExternalValue) Store(v Value) {
	p.Object.Value.Elem().Set(v.RValue())
}

// ToValue 返回外部变量对应的值
func (p *ExternalValue) ToValue() Value {
	return RValue{p.Object.Value}
}

// Interface 返回外部变量的实际值
func (p *ExternalValue) Interface() interface{} {
	return p.Object
}

// RValue 反射类型的变量
type RValue struct {
	reflect.Value
}

// RValue 实现Value接口
func (p RValue) RValue() reflect.Value {
	return p.Value
}

// Next 实现Value接口
func (p RValue) Next() Value {
	panic("implement")
}

// Field 实现Value接口
func (p RValue) Field(i int) Value {
	return RValue{p.Value.Field(i)}
}

// MapIndex 实现Value接口
func (p RValue) MapIndex(v Value) Value {
	return RValue{p.Value.MapIndex(v.RValue())}
}

// Set 实现Value接口
func (p RValue) Set(v Value) {
	p.Value.Set(v.RValue())
}

// Index 实现Value接口
func (p RValue) Index(i int) Value {
	return RValue{p.Value.Index(i)}
}

// Elem 实现Value接口
func (p RValue) Elem() Value {
	if p.Value.Kind() == reflect.Ptr || p.Value.Kind() == reflect.Interface {
		return RValue{p.Value.Elem()}
	}
	return p
}

// IsNil 实现Value接口实现Value接口
func (p RValue) IsNil() bool {
	switch p.Kind() {
	case reflect.Chan, reflect.Func, reflect.Map, reflect.Ptr, reflect.UnsafePointer, reflect.Interface, reflect.Slice:
		return p.Value.IsNil()
	default:
		return false
	}
}

// MapIter map迭代器
type MapIter struct {
	I int
	Value
	Keys []reflect.Value
}

// Next 取map迭代器的下一个值
func (p *MapIter) Next() Value {
	v := make([]Value, 3)
	if p.I < len(p.Keys) {
		k := RValue{p.Keys[p.I]}
		v[0] = ValueOf(true)
		v[1] = k
		v[2] = p.MapIndex(k)
		p.I++
	} else {
		v[0] = ValueOf(false)
	}
	return ValueOf(v)
}

// ValueOf 将v转换为gofun中的值
func ValueOf(v interface{}) Value {
	return RValue{reflect.ValueOf(v)}
}

// Package 将多值打包为单个Value
func Package(values []reflect.Value) Value {
	l := len(values)
	switch l {
	case 0:
		return nil
	case 1:
		return RValue{values[0]}
	default:
		v := make([]Value, l)
		for i := range v {
			v[i] = RValue{values[i]}
		}
		return ValueOf(v)
	}
}

// Unpackage 将单个Value解包成Value数组
func Unpackage(val Value) []reflect.Value {
	if val == nil {
		return nil
	}
	if arr, ok := val.Interface().([]Value); ok {
		ret := make([]reflect.Value, len(arr))
		for i, v := range arr {
			ret[i] = v.RValue()
		}
		return ret
	}
	return []reflect.Value{val.RValue()}
}

// ExternalValueWrap 遍历语句中的所有操作数，对外部变量进行替换
func ExternalValueWrap(p *importer.Importer, pkg *ssa.Package) {
	for f := range ssautil.AllFunctions(pkg.Prog) {
		for _, block := range f.Blocks {
			for _, instr := range block.Instrs {
				for _, v := range instr.Operands(nil) {
					valueWrap(p, v)
				}
			}
		}
	}
}

func valueWrap(p *importer.Importer, v *ssa.Value) {
	if *v == nil {
		return
	}
	name := strings.TrimLeft((*v).String(), "*&")
	dotIndex := strings.IndexRune(name, '.')
	if dotIndex < 0 {
		return
	}
	pkgName := name[:dotIndex]
	if pkg := p.SsaPackage(pkgName); pkg != nil {
		if value, ok := pkg.Members[name[dotIndex+1:]].(ssa.Value); ok {
			*v = value
			return
		}
	}
	// 导入外部变量数据
	if external := p.ExternalObject(name); external != nil {
		*v = &ExternalValue{
			Value:  *v,
			Object: external,
		}
		return
	}
}
