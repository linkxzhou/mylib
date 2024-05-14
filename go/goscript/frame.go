package gofun

import (
	"context"
	"fmt"
	"reflect"
	"strings"
	"time"

	"git.woa.com/vasd_masc_ba/YitihuaOteam/base/gofun/internal/value"
	"golang.org/x/tools/go/ssa"
)

const defaultTimeout = 10 * time.Second

// Context 函数执行的上下文环境
type Context struct {
	context.Context
	outBuffer  strings.Builder
	goroutines int32
	cancelFunc context.CancelFunc
}

// Output 返回内置函数print和println打印的内容
func (p *Context) Output() string {
	return p.outBuffer.String()
}

func newCallContext() *Context {
	ctx, cancelFunc := context.WithTimeout(context.Background(), defaultTimeout)
	return &Context{
		Context:    ctx,
		cancelFunc: cancelFunc,
	}
}

// frame 函数栈帧，每次调用函数都会生成一个新的栈帧，用于存储该函数返回地址、参数、局部变量等信息
type frame struct {
	program          *Program
	caller           *frame
	fn               *ssa.Function
	block, prevBlock *ssa.BasicBlock
	env              map[ssa.Value]*value.Value
	locals           []value.Value
	defers           []*ssa.Defer
	result           value.Value
	panicking        bool
	panic            interface{}

	context *Context
}

// makeFunc 定义函数或创建闭包，bindings为闭包中关联的外部变量
func (fr *frame) makeFunc(f *ssa.Function, bindings []ssa.Value) value.Value {
	env := make([]*value.Value, len(bindings))
	for i, binding := range bindings {
		env[i] = fr.env[binding]
	}
	in := make([]reflect.Type, len(f.Params))
	for i, param := range f.Params {
		in[i] = typeChange(param.Type())
	}
	out := make([]reflect.Type, 0)
	results := f.Signature.Results()
	for i := 0; i < results.Len(); i++ {
		out = append(out, typeChange(results.At(i).Type()))
	}
	funcType := reflect.FuncOf(in, out, f.Signature.Variadic())
	fn := func(in []reflect.Value) (results []reflect.Value) {
		args := make([]value.Value, len(in))
		for i, arg := range in {
			args[i] = value.RValue{Value: arg}
		}
		ret := callSSA(fr, f, args, env)
		if ret != nil {
			return value.Unpackage(ret)
		} else {
			return nil
		}
	}
	return value.RValue{Value: reflect.MakeFunc(funcType, fn)}
}

func (fr *frame) get(key ssa.Value) value.Value {
	switch key := key.(type) {
	case nil:
		return nil
	case *ssa.Const:
		return constValue(key)
	case *ssa.Global:
		if r, ok := fr.program.globals[key]; ok {
			v := (*r).Interface()
			return value.ValueOf(&v)
		}
	case *value.ExternalValue:
		return key.ToValue()
	case *ssa.Function:
		return fr.makeFunc(key, nil)
	}
	if r, ok := fr.env[key]; ok {
		return *r
	}
	panic(fmt.Sprintf("get: no Value for %T: %v", key, key.Name()))
}

func (fr *frame) set(instr ssa.Value, value value.Value) {
	fr.env[instr] = &value
}

func (fr *frame) newChild(fn *ssa.Function) *frame {
	return &frame{
		program: fr.program,
		context: fr.context,
		caller:  fr, // for panic/recover
		fn:      fn,
	}
}

func (fr *frame) runDefers() {
	for i := len(fr.defers) - 1; i >= 0; i-- {
		fr.runDefer(fr.defers[i])
	}
	fr.defers = nil
	if fr.panicking {
		panic(fr.panic)
	}
}

func (fr *frame) runDefer(d *ssa.Defer) {
	var ok bool
	defer func() {
		if !ok {
			fr.panicking = true
			fr.panic = recover()
		}
	}()
	callOp(fr, d.Common())
	ok = true
}
