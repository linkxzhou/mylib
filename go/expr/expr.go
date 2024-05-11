package expr

import (
	"errors"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"reflect"
	"strconv"
	"strings"
	"sync"
)

var (
	ErrUnsupportedType = errors.New("unsupported type")
	ErrDivideZero      = errors.New("divide zero")
)

type (
	Kind   uint8
	OpKind uint8

	Int   int64
	Float float64
	Raw   []byte
	Bool  bool
)

const (
	kInvalid Kind = 1 << iota // 1
	kBool                     // 2
	kInt                      // 4
	kFloat                    // 8
	kRaw                      // 16
)

const (
	OpKindAdd OpKind = iota
	OpKindSub
	OpKindMul
	OpKindQuo
	OpKindRem
	OpKindPow
	OpKindOr
	OpKindNot
	OpKindEq
	OpKindNe
	OpKindGt
	OpKindGe
	OpKindLt
	OpKindLe
)

type (
	Value interface {
		Kind() Kind
		Add(v2 Value) (Value, error)
		Sub(v2 Value) (Value, error)
		Mul(v2 Value) (Value, error)
		Quo(v2 Value) (Value, error)
		Rem(v2 Value) (Value, error)
		And(v2 Value) (Value, error)
		Or(v2 Value) (Value, error)
		Shl(v2 Value) (Value, error)
		Shr(v2 Value) (Value, error)
		AndNot(v2 Value) (Value, error)
		Xor() (Value, error)
		Land(v2 Value) (Value, error)
		Lor(v2 Value) (Value, error)
		Not() (Value, error)
		Eq(v2 Value) (Value, error)
		Ne(v2 Value) (Value, error)
		Gt(v2 Value) (Value, error)
		Ge(v2 Value) (Value, error)
		Lt(v2 Value) (Value, error)
		Le(v2 Value) (Value, error)

		Int() int64
		Float() float64
		String() string
		Bool() bool
	}

	InvalidValue struct{}
)

func NewValue(v interface{}) (Value, error) {
	switch reflect.TypeOf(v).Kind() {
	case reflect.Int:
		return Int(v.(int)), nil
	case reflect.Int16:
		return Int(v.(int16)), nil
	case reflect.Int32:
		return Int(v.(int32)), nil
	case reflect.Int64:
		return Int(v.(int64)), nil
	case reflect.Float32:
		return Float(v.(float32)), nil
	case reflect.Float64:
		return Float(v.(float64)), nil
	case reflect.String:
		return Raw(v.(string)), nil
	}
	return nil, ErrUnsupportedType
}

func (k Kind) String() string {
	switch k {
	case kInvalid:
		return "invalid"
	case kBool:
		return "bool"
	case kInt:
		return "int"
	case kFloat:
		return "float"
	case kRaw:
		return "raw"
	}
	return "undefined"
}

func support(v Value, kind Kind) error {
	if v.Kind()&kind != v.Kind() {
		return fmt.Errorf("unsupported type %v", v.Kind().String())
	}
	return nil
}

func (v Bool) Int() int64     { return 0 }
func (v Bool) Float() float64 { return 0 }
func (v Bool) String() string { return fmt.Sprintf("%v", bool(v)) }
func (v Bool) Bool() bool     { return bool(v) }
func (v Bool) Kind() Kind     { return kBool }

func (v Bool) Add(v2 Value) (Value, error)    { return nil, ErrUnsupportedType }
func (v Bool) Sub(v2 Value) (Value, error)    { return nil, ErrUnsupportedType }
func (v Bool) Mul(v2 Value) (Value, error)    { return nil, ErrUnsupportedType }
func (v Bool) Quo(v2 Value) (Value, error)    { return nil, ErrUnsupportedType }
func (v Bool) Rem(v2 Value) (Value, error)    { return nil, ErrUnsupportedType }
func (v Bool) And(v2 Value) (Value, error)    { return nil, ErrUnsupportedType }
func (v Bool) Or(v2 Value) (Value, error)     { return nil, ErrUnsupportedType }
func (v Bool) Shl(v2 Value) (Value, error)    { return nil, ErrUnsupportedType }
func (v Bool) Shr(v2 Value) (Value, error)    { return nil, ErrUnsupportedType }
func (v Bool) AndNot(v2 Value) (Value, error) { return nil, ErrUnsupportedType }
func (v Bool) Xor() (Value, error)            { return nil, ErrUnsupportedType }
func (v Bool) Land(v2 Value) (Value, error)   { return Bool(v.Bool() && v2.Bool()), nil }
func (v Bool) Lor(v2 Value) (Value, error)    { return Bool(v.Bool() || v2.Bool()), nil }
func (v Bool) Not() (Value, error)            { return Bool(!v.Bool()), nil }
func (v Bool) Eq(v2 Value) (Value, error)     { return nil, ErrUnsupportedType }
func (v Bool) Ne(v2 Value) (Value, error)     { return nil, ErrUnsupportedType }
func (v Bool) Gt(v2 Value) (Value, error)     { return nil, ErrUnsupportedType }
func (v Bool) Ge(v2 Value) (Value, error)     { return nil, ErrUnsupportedType }
func (v Bool) Lt(v2 Value) (Value, error)     { return nil, ErrUnsupportedType }
func (v Bool) Le(v2 Value) (Value, error)     { return nil, ErrUnsupportedType }

func (v Int) Int() int64     { return int64(v) }
func (v Int) Float() float64 { return float64(v) }
func (v Int) String() string { return fmt.Sprintf("%v", int64(v)) }
func (v Int) Bool() bool     { return v != 0 }
func (v Int) Kind() Kind     { return kInt }

func (v Int) Add(v2 Value) (Value, error) { return Int(int64(v) + v2.Int()), support(v, kInt|kFloat) }
func (v Int) Sub(v2 Value) (Value, error) { return Int(int64(v) - v2.Int()), support(v, kInt|kFloat) }
func (v Int) Mul(v2 Value) (Value, error) { return Int(int64(v) * v2.Int()), support(v, kInt|kFloat) }
func (v Int) Quo(v2 Value) (Value, error) {
	if v2.Int() == 0 {
		return nil, ErrDivideZero
	}
	return Int(int64(v) / v2.Int()), support(v, kInt|kFloat)
}
func (v Int) Rem(v2 Value) (Value, error) { return Int(int64(v) % v2.Int()), support(v, kInt|kFloat) }
func (v Int) And(v2 Value) (Value, error) { return Int(int64(v) & v2.Int()), support(v, kInt|kFloat) }
func (v Int) Or(v2 Value) (Value, error)  { return Int(int64(v) | v2.Int()), support(v, kInt|kFloat) }
func (v Int) Shl(v2 Value) (Value, error) { return Int(int64(v) << v2.Int()), support(v, kInt|kFloat) }
func (v Int) Shr(v2 Value) (Value, error) { return Int(int64(v) >> v2.Int()), support(v, kInt|kFloat) }
func (v Int) AndNot(v2 Value) (Value, error) {
	return Int(int64(v) &^ v2.Int()), support(v, kInt|kFloat)
}
func (v Int) Xor() (Value, error)          { return Int(^int64(v)), nil }
func (v Int) Land(v2 Value) (Value, error) { return Bool(v.Bool() && v2.Bool()), nil }
func (v Int) Lor(v2 Value) (Value, error)  { return Bool(v.Bool() || v2.Bool()), nil }
func (v Int) Not() (Value, error)          { return Bool(!v.Bool()), nil }
func (v Int) Eq(v2 Value) (Value, error)   { return Bool(int64(v) == v2.Int()), support(v, kInt|kFloat) }
func (v Int) Ne(v2 Value) (Value, error)   { return Bool(int64(v) != v2.Int()), support(v, kInt|kFloat) }
func (v Int) Gt(v2 Value) (Value, error) {
	return Bool(float64(v) > v2.Float()), support(v, kInt|kFloat)
}
func (v Int) Ge(v2 Value) (Value, error) {
	return Bool(float64(v) >= v2.Float()), support(v, kInt|kFloat)
}
func (v Int) Lt(v2 Value) (Value, error) {
	return Bool(float64(v) < v2.Float()), support(v, kInt|kFloat)
}
func (v Int) Le(v2 Value) (Value, error) {
	return Bool(float64(v) <= v2.Float()), support(v, kInt|kFloat)
}

func (v Float) Int() int64     { return int64(v) }
func (v Float) Float() float64 { return float64(v) }
func (v Float) String() string { return fmt.Sprintf("%v", float64(v)) }
func (v Float) Bool() bool     { return v != 0 }
func (v Float) Kind() Kind     { return kFloat }

func (v Float) Add(v2 Value) (Value, error) {
	return Float(float64(v) + v2.Float()), support(v, kInt|kFloat)
}
func (v Float) Sub(v2 Value) (Value, error) {
	return Float(float64(v) - v2.Float()), support(v, kInt|kFloat)
}
func (v Float) Mul(v2 Value) (Value, error) {
	return Float(float64(v) * v2.Float()), support(v, kInt|kFloat)
}
func (v Float) Quo(v2 Value) (Value, error) {
	return Float(float64(v) / v2.Float()), support(v, kInt|kFloat)
}
func (v Float) Rem(v2 Value) (Value, error)    { return nil, ErrUnsupportedType }
func (v Float) And(v2 Value) (Value, error)    { return nil, ErrUnsupportedType }
func (v Float) Or(v2 Value) (Value, error)     { return nil, ErrUnsupportedType }
func (v Float) Shl(v2 Value) (Value, error)    { return nil, ErrUnsupportedType }
func (v Float) Shr(v2 Value) (Value, error)    { return nil, ErrUnsupportedType }
func (v Float) AndNot(v2 Value) (Value, error) { return nil, ErrUnsupportedType }
func (v Float) Xor() (Value, error)            { return nil, ErrUnsupportedType }
func (v Float) Land(v2 Value) (Value, error)   { return nil, ErrUnsupportedType }
func (v Float) Lor(v2 Value) (Value, error)    { return nil, ErrUnsupportedType }
func (v Float) Not() (Value, error)            { return nil, ErrUnsupportedType }
func (v Float) Eq(v2 Value) (Value, error) {
	return Bool(float64(v) == v2.Float()), support(v, kInt|kFloat)
}
func (v Float) Ne(v2 Value) (Value, error) {
	return Bool(float64(v) != v2.Float()), support(v, kInt|kFloat)
}
func (v Float) Gt(v2 Value) (Value, error) {
	return Bool(float64(v) > v2.Float()), support(v, kInt|kFloat)
}
func (v Float) Ge(v2 Value) (Value, error) {
	return Bool(float64(v) >= v2.Float()), support(v, kInt|kFloat)
}
func (v Float) Lt(v2 Value) (Value, error) {
	return Bool(float64(v) < v2.Float()), support(v, kInt|kFloat)
}
func (v Float) Le(v2 Value) (Value, error) {
	return Bool(float64(v) <= v2.Float()), support(v, kInt|kFloat)
}

func (v Raw) Int() int64     { return 0 }
func (v Raw) Float() float64 { return 0 }
func (v Raw) String() string { return fmt.Sprintf("%v", string(v)) }
func (v Raw) Bool() bool     { return string(v) != "" }
func (v Raw) Kind() Kind     { return kRaw }

func (v Raw) Add(v2 Value) (Value, error) {
	return Raw(string(v) + v2.String()), support(v, kRaw)
}
func (v Raw) Sub(v2 Value) (Value, error)    { return nil, ErrUnsupportedType }
func (v Raw) Mul(v2 Value) (Value, error)    { return nil, ErrUnsupportedType }
func (v Raw) Quo(v2 Value) (Value, error)    { return nil, ErrUnsupportedType }
func (v Raw) Rem(v2 Value) (Value, error)    { return nil, ErrUnsupportedType }
func (v Raw) And(v2 Value) (Value, error)    { return nil, ErrUnsupportedType }
func (v Raw) Or(v2 Value) (Value, error)     { return nil, ErrUnsupportedType }
func (v Raw) Shl(v2 Value) (Value, error)    { return nil, ErrUnsupportedType }
func (v Raw) Shr(v2 Value) (Value, error)    { return nil, ErrUnsupportedType }
func (v Raw) AndNot(v2 Value) (Value, error) { return nil, ErrUnsupportedType }
func (v Raw) Xor() (Value, error)            { return nil, ErrUnsupportedType }
func (v Raw) Land(v2 Value) (Value, error)   { return nil, ErrUnsupportedType }
func (v Raw) Lor(v2 Value) (Value, error)    { return nil, ErrUnsupportedType }
func (v Raw) Not() (Value, error)            { return nil, ErrUnsupportedType }
func (v Raw) Eq(v2 Value) (Value, error) {
	return Bool(string(v) == v2.String()), support(v, kRaw)
}
func (v Raw) Ne(v2 Value) (Value, error) {
	return Bool(string(v) != v2.String()), support(v, kRaw)
}
func (v Raw) Gt(v2 Value) (Value, error) {
	return Bool(string(v) > v2.String()), support(v, kRaw)
}
func (v Raw) Ge(v2 Value) (Value, error) {
	return Bool(string(v) >= v2.String()), support(v, kRaw)
}
func (v Raw) Lt(v2 Value) (Value, error) {
	return Bool(string(v) < v2.String()), support(v, kRaw)
}
func (v Raw) Le(v2 Value) (Value, error) {
	return Bool(string(v) <= v2.String()), support(v, kRaw)
}

func ValuesToInterfaces(args ...Value) []interface{} {
	var rInterfaces []interface{}
	for _, v := range args {
		switch v.Kind() {
		case kBool:
			rInterfaces = append(rInterfaces, v.Bool())
		case kInt:
			rInterfaces = append(rInterfaces, v.Int())
		case kFloat:
			rInterfaces = append(rInterfaces, v.Float())
		case kRaw:
			rInterfaces = append(rInterfaces, v.String())
		}
	}
	return rInterfaces
}

type (
	BuiltinFunc func(...interface{}) (interface{}, error)

	Expr struct {
		root  ast.Expr
		pool  *Pool
		cache map[string]Value
	}

	Getter map[string]interface{}
)

func (getter Getter) get(name string) (Value, bool) {
	if getter == nil {
		return nil, false
	}

	v, ok := getter[name]
	if !ok {
		return nil, false
	}

	rValue, err := NewValue(v)
	return rValue, err == nil
}

var (
	defaultPool = func() *Pool {
		p, err := NewPool()
		if err != nil {
			panic(err)
		}
		return p
	}()

	defaultOnVarMissing = func(varName string) (Value, error) {
		return nil, fmt.Errorf("var `%s' missing", varName)
	}
)

func New(s string, pool *Pool) (*Expr, error) {
	s = strings.TrimSpace(s)
	if pool == nil {
		pool = defaultPool
	}
	if e, ok := pool.get(s); ok {
		return e, nil
	}
	e := new(Expr)
	e.pool = pool
	e.cache = make(map[string]Value, 0)
	if err := e.parse(s); err != nil {
		return nil, err
	}
	pool.set(s, e)
	return e, nil
}

// parse parses string s
func (e *Expr) parse(s string) error {
	if s == "" {
		return fmt.Errorf("parse error: empty string")
	}
	node, err := parser.ParseExpr(s)
	if err != nil {
		return err
	}
	e.root = node
	v := &visitor{pool: e.pool}
	ast.Walk(v, e.root)
	return v.err
}

type visitor struct {
	pool *Pool
	err  error
}

// Visit implements ast.Visitor Visit method
func (v *visitor) Visit(node ast.Node) ast.Visitor {
	if n, ok := node.(*ast.CallExpr); ok {
		if fnIdent, ok := n.Fun.(*ast.Ident); ok {
			if _, ok := v.pool.builtinList[fnIdent.Name]; !ok {
				v.err = fmt.Errorf("undefined function `%v`", fnIdent.Name)
			}
		} else {
			v.err = fmt.Errorf("unsupported call expr")
		}
	}
	return v
}

func (e *Expr) Eval(getter Getter) (Value, error) {
	if e.root == nil {
		return nil, nil
	}
	v, err := eval(e, getter, e.root)
	if err != nil {
		return nil, err
	}
	return v, nil
}

func Eval(s string, getter Getter, pool *Pool) (Value, error) {
	e, err := New(s, pool)
	if err != nil {
		return nil, err
	}
	return e.Eval(Getter(getter))
}

func eval(e *Expr, getter Getter, node ast.Expr) (Value, error) {
	switch n := node.(type) {
	case *ast.Ident:
		// support true/false
		switch strings.ToLower(n.Name) {
		case "true":
			return Bool(true), nil
		case "false":
			return Bool(false), nil
		}

		if getter == nil {
			return e.pool.onVarMissing(n.Name)
		}

		if val, ok := getter.get(n.Name); !ok {
			return e.pool.onVarMissing(n.Name)
		} else {
			return val, nil
		}

	case *ast.BasicLit:
		switch n.Kind {
		case token.INT:
			if v, ok := e.cache[n.Value]; ok {
				return v, nil
			}
			i, err := strconv.ParseInt(n.Value, 10, 64)
			if err != nil {
				return nil, err
			}
			v := Int(i)
			e.cache[n.Value] = v
			return v, nil
		case token.FLOAT:
			if v, ok := e.cache[n.Value]; ok {
				return v, nil
			}
			f, err := strconv.ParseFloat(n.Value, 64)
			if err != nil {
				return nil, err
			}
			v := Float(f)
			e.cache[n.Value] = v
			return v, nil
		case token.CHAR, token.STRING:
			if v, ok := e.cache[n.Value]; ok {
				return v, nil
			}
			s, err := strconv.Unquote(n.Value)
			if err != nil {
				return nil, err
			}
			v := Raw(s)
			e.cache[n.Value] = v
			return v, nil
		default:
			return nil, fmt.Errorf("unsupported token: %s(%v)", n.Value, n.Kind)
		}

	case *ast.ParenExpr:
		return eval(e, getter, n.X)

	case *ast.CallExpr:
		args := make([]Value, 0, len(n.Args))
		for _, arg := range n.Args {
			val, err := eval(e, getter, arg)
			if err != nil {
				return nil, err
			}
			args = append(args, val)
		}
		if fnIdent, ok := n.Fun.(*ast.Ident); ok {
			return e.pool.builtinCall(fnIdent.Name, args...)
		}
		return nil, fmt.Errorf("unexpected func type: %T", n.Fun)

	case *ast.UnaryExpr:
		switch n.Op {
		case token.ADD:
			return eval(e, getter, n.X)
		case token.SUB:
			return eval(e, getter, n.X)
		case token.NOT:
			x, err := eval(e, getter, n.X)
			if err == nil {
				x, err = x.Not()
			}
			return x, err
		default:
			return nil, fmt.Errorf("unsupported unary op: %v", n.Op)
		}

	case *ast.BinaryExpr:
		x, err := eval(e, getter, n.X)
		if err != nil {
			return nil, err
		}
		y, err := eval(e, getter, n.Y)
		if err != nil {
			return nil, err
		}
		switch n.Op {
		case token.ADD:
			return x.Add(y)
		case token.SUB:
			return x.Sub(y)
		case token.MUL:
			return x.Mul(y)
		case token.QUO:
			return x.Quo(y)
		case token.REM:
			return x.Rem(y)
		case token.XOR:
			return x.Xor()
		case token.OR:
			return x.Or(y)
		case token.AND:
			return x.And(y)
		case token.SHL:
			return x.Shl(y)
		case token.SHR:
			return x.Shr(y)
		case token.AND_NOT:
			return x.AndNot(y)
		case token.LAND:
			return x.Land(y)
		case token.LOR:
			return x.Lor(y)
		case token.EQL:
			return x.Eq(y)
		case token.NEQ:
			return x.Ne(y)
		case token.GTR:
			return x.Gt(y)
		case token.GEQ:
			return x.Ge(y)
		case token.LSS:
			return x.Lt(y)
		case token.LEQ:
			return x.Le(y)
		default:
			return nil, fmt.Errorf("unexpected binary operator: %v", n.Op)
		}

	default:
		return nil, fmt.Errorf("unexpected node type %v", n)
	}
}

type (
	VarMissingFunc func(string) (Value, error)

	Pool struct {
		locker sync.RWMutex
		pool   map[string]*Expr

		builtinList  map[string]BuiltinFunc
		onVarMissing VarMissingFunc
	}
)

func NewPool(builtinList ...map[string]BuiltinFunc) (*Pool, error) {
	p := &Pool{
		pool:         make(map[string]*Expr),
		builtinList:  map[string]BuiltinFunc{},
		onVarMissing: defaultOnVarMissing,
	}
	for _, builtin := range builtinList {
		for name, fn := range builtin {
			p.builtinList[name] = fn
		}
	}
	return p, nil
}

func (p *Pool) SetOnVarMissing(fn VarMissingFunc) {
	p.onVarMissing = fn
}

func (p *Pool) get(s string) (*Expr, bool) {
	p.locker.RLock()
	defer p.locker.RUnlock()
	e, ok := p.pool[s]
	return e, ok && e != nil
}

func (p *Pool) set(s string, e *Expr) {
	p.locker.Lock()
	defer p.locker.Unlock()
	p.pool[s] = e
}

func (p *Pool) builtinCall(name string, args ...Value) (Value, error) {
	if fn, ok := p.builtinList[name]; ok {
		v, err := fn(ValuesToInterfaces(args...)...)
		if err == nil {
			return NewValue(v)
		}
		return nil, err
	}
	return nil, fmt.Errorf("undefined function `%v`", name)
}
