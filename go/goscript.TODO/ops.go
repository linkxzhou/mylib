package gofun

import (
	"fmt"
	"go/constant"
	"go/token"
	"go/types"
	"reflect"
	"sync/atomic"

	"git.woa.com/vasd_masc_ba/YitihuaOteam/base/gofun/internal/value"
	"golang.org/x/tools/go/ssa"
)

// upop 一元表达式求值
func unop(instr *ssa.UnOp, x value.Value) value.Value {
	if instr.Op == token.MUL {
		return value.ValueOf(x.Elem().Interface())
	}
	var result interface{}
	switch x.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		switch instr.Op {
		case token.SUB:
			result = -x.Int()
		case token.XOR:
			result = ^x.Int()
		default:
			panic(fmt.Sprintf("invalid unary op %s %T", instr.Op, x))
		}
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		switch instr.Op {
		case token.SUB:
			result = -x.Uint()
		case token.XOR:
			result = ^x.Uint()
		default:
			panic(fmt.Sprintf("invalid unary op %s %T", instr.Op, x))
		}
	case reflect.Float32, reflect.Float64, reflect.Complex64, reflect.Complex128:
		switch instr.Op {
		case token.SUB:
			result = -x.Float()
		default:
			panic(fmt.Sprintf("invalid unary op %s %T", instr.Op, x))
		}
	case reflect.Bool:
		switch instr.Op {
		case token.NOT:
			result = !x.Bool()
		default:
			panic(fmt.Sprintf("invalid unary op %s %T", instr.Op, x))
		}
	case reflect.Chan: // receive
		v, ok := x.RValue().Recv()
		if !ok {
			v = reflect.Zero(x.Type().Elem())
		}
		if instr.CommaOk {
			return value.ValueOf([]value.Value{value.RValue{Value: v}, value.ValueOf(ok)})
		}
		return value.RValue{Value: v}
	}
	return conv(result, instr.Type())
}

// constValue 常量表达式求值
func constValue(c *ssa.Const) value.Value {
	if c.IsNil() {
		return zero(c.Type()).Elem() // typed nil
	}
	var val interface{}
	t := c.Type().Underlying().(*types.Basic)
	switch t.Kind() {
	case types.Bool, types.UntypedBool:
		val = constant.BoolVal(c.Value)
	case types.Int, types.UntypedInt, types.Int8, types.Int16, types.Int32, types.UntypedRune, types.Int64:
		val = c.Int64()
	case types.Uint, types.Uint8, types.Uint16, types.Uint32, types.Uint64, types.Uintptr:
		val = c.Uint64()
	case types.Float32, types.Float64, types.UntypedFloat:
		val = c.Float64()
	case types.Complex64, types.Complex128, types.UntypedComplex:
		val = c.Complex128()
	case types.String, types.UntypedString:
		if c.Value.Kind() == constant.String {
			val = constant.StringVal(c.Value)
		} else {
			val = string(rune(c.Int64()))
		}
	default:
		panic(fmt.Sprintf("constValue: %s", c))
	}
	return conv(val, c.Type())
}

// binop 二元表达式求值
// nolint:gocognit,gocyclo,funlen
func binop(instr *ssa.BinOp, x, y value.Value) value.Value {
	var result interface{}
	switch instr.Op {
	case token.ADD: // +
		result = binopADD(x, y)

	case token.SUB: // -
		result = binopSUB(x, y)

	case token.MUL: // *
		result = binopMUL(x, y)

	case token.QUO: // /
		result = binopQUO(x, y)

	case token.REM: // %
		result = binopREM(x, y)

	case token.AND: // &
		result = binopAND(x, y)

	case token.OR: // |
		result = binopOR(x, y)

	case token.XOR: // ^
		result = binopXOR(x, y)

	case token.AND_NOT: // &^
		result = binopANDNOT(x, y)

	case token.SHL: // <<
		result = binopSHL(x, y)

	case token.SHR: // >>
		result = binopSHR(x, y)

	case token.LSS: // <
		result = binopLSS(x, y)

	case token.LEQ: // <=
		result = binopLEQ(x, y)

	case token.EQL: // ==
		result = binopEQL(x, y)

	case token.NEQ: // !=
		result = binopNEQ(x, y)

	case token.GTR: // >
		result = binopGTR(x, y)

	case token.GEQ: // >=
		result = binopGEQ(x, y)
	}
	return conv(result, instr.Type())
}

// goCall go语句执行
func goCall(fr *frame, instr *ssa.CallCommon) {
	if instr.Signature().Recv() != nil {
		recv := fr.get(instr.Args[0])
		if recv.RValue().NumMethod() > 0 { // external method
			args := make([]value.Value, len(instr.Args)-1)
			for i := range args {
				args[i] = fr.get(instr.Args[i+1])
			}
			go callExternal(recv.RValue().MethodByName(instr.Value.Name()), args)
			return
		}
	}

	args := make([]value.Value, len(instr.Args))
	for i, arg := range instr.Args {
		args[i] = fr.get(arg)
	}

	atomic.AddInt32(&fr.context.goroutines, 1)

	go func(caller *frame, fn ssa.Value, args []value.Value) {
		defer func() {
			// 启动协程前添加recover语句，避免协程panic影响其他协程
			re := recover()
			if re != nil {
				caller.context.outBuffer.WriteString(fmt.Sprintf("goroutine panic: %v", re))
			}
			atomic.AddInt32(&caller.context.goroutines, -1)
		}()
		call(caller, instr.Pos(), fn, args)
	}(fr, instr.Value, args)
}

// callOp 函数调用语句执行
func callOp(fr *frame, instr *ssa.CallCommon) value.Value {
	if instr.Signature().Recv() == nil {
		// call func
		args := make([]value.Value, len(instr.Args))
		for i, arg := range instr.Args {
			args[i] = fr.get(arg)
		}
		return call(fr, instr.Pos(), instr.Value, args)
	}

	// invoke Method
	if instr.IsInvoke() {
		recv := fr.get(instr.Value)
		args := make([]value.Value, len(instr.Args))
		for i := range args {
			args[i] = fr.get(instr.Args[i])
		}
		return callExternal(recv.RValue().MethodByName(instr.Method.Name()), args)
	}

	args := make([]value.Value, len(instr.Args))
	for i, arg := range instr.Args {
		args[i] = fr.get(arg)
	}
	if args[0].Type().NumMethod() == 0 {
		return call(fr, instr.Pos(), instr.Value, args)
	}
	return callExternal(args[0].RValue().MethodByName(instr.Value.Name()), args[1:])
}

// call 函数调用
func call(caller *frame, callpos token.Pos, fn interface{}, args []value.Value) value.Value {
	switch fun := fn.(type) {
	case *ssa.Function:
		if fun == nil {
			panic("call of nil function") // nil of func type
		}
		return callSSA(caller, fun, args, nil)
	case *ssa.Builtin:
		return callBuiltin(caller, callpos, fun, args)
	case *value.ExternalValue:
		return callExternal(fun.Object.Value, args)
	case ssa.Value:
		p := caller.env[fun]
		f := (*p).Interface()
		return call(caller, callpos, f, args)
	default:
		return callExternal(reflect.ValueOf(fun), args)
	}
}
