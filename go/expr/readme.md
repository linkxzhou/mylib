# 开源项目|500行代码实现表达式引擎  

## 1、如何在golang实现表达式引擎？   

### 1.1、什么是表达式引擎？   
表达式引擎，顾名思义就是将表达式解析成可执行的代码，比如：`1+2`，`1+2*3`，`(1+2)*3`，`"hello " + "world"`等，更复杂的表达式，比如：`sum(1+2)+1+1/2.0`等。     
在golang中为什么要表达式引擎？由于golang是静态语言，无法像其他语言一样动态执行表达式，比如：`1+2`，`"hello " + "world"`等，那么就需要将表达式解析成可执行的代码。     

**使用场景有哪些呢？**     
- 通用计算器，比如：`1+2*3`，`(1+2)*3`等；    
- 规则引擎，比如：`1+2*3>5`，`(1+2)*3>5`等；     
- 通过表达式引擎判断某些动态加载的代码是否合法；      
- ...

**支持表达式符号**  
- 算术运算符：`+ - * / % ^ & | << >> && || !`    
- 逻辑运算符：`== != > >= < <= && || !`    
- 字符串运算符：`+`  
- 函数调用，可以自定义函数，如：`sum(1,2)`  
- 支持变量，如：`a=1,b=2,c=3`  

### 1.2、通过 `AST` 实现

实现表达式引擎的方式有很多，大部分还是使用 `AST`实现，大体流程：

- 对表达式进行词法解析  
- 生成对应的token序列  
- 然后进行语义分析  
- 生成 `AST` 节点  
- 对 `AST` 节点进行遍历，执行对应 `AST` 节点操作  

### 1.3、核心代码

**（1）解析生成AST**   
解析生成 `AST` 在golang中，可以使用 `go/ast` 包实现，具体代码：    
- parser.ParseExpr 解析表达式的字符串，生成 `ast.Expr` 节点
- ast.Walk 遍历 `ast.Expr` 节点，如果节点中包含函数调用，则判断是否为内置函数，如果不是内置函数则报错 

```go
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
```

**（2）递归遍历`ast.Expr`**

- 遍历 `ast.Expr` 节点，判断节点的类型
- 对于一些支持的操作符，比如：`+ - * / % ^ & | << >> && || !`，则递归遍历左右节点
- 对于 `ast.CallExpr` 节点，则判断是否为内置函数，如果不是内置函数则报错
- 核心代码如下：  
```go
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

		v, ok := getter.get(n.Name)
		if !ok {
			return e.pool.onVarMissing(n.Name)
		}
		return v, nil

	case *ast.BasicLit:
		pos := int64(n.ValuePos)
		var v Value
		if e.cacheValues != nil {
			if v, ok := e.cacheValues[pos]; ok {
				return v, nil
			}
		}

		switch n.Kind {
		case token.INT:
			i, err := strconv.ParseInt(n.Value, 10, 64)
			if err != nil {
				return nil, err
			}
			v = Int(i)
		case token.FLOAT:
			f, err := strconv.ParseFloat(n.Value, 64)
			if err != nil {
				return nil, err
			}
			v = Float(f)
		case token.CHAR, token.STRING:
			s, err := strconv.Unquote(n.Value)
			if err != nil {
				return nil, err
			}
			v = Raw(s)
		default:
			return nil, fmt.Errorf("unsupported token: %s(%v)", n.Value, n.Kind)
		}
		if e.cacheValues != nil {
			e.cacheValues[pos] = v
		}
		return v, nil

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
```

## 2、使用方式   

开源地址：https://github.com/linkxzhou/mylib.git  

### 2.1、基本用法     

- 根据表达式，New一个对象    
- 调用Eval方法，传入变量值，如果没有变量值，则传入nil     

```go  
func main() {
    const evalstr = `((2 << 3) + (10 % 3)) * (5 - (x * 2)) + (3.0 / y) * (2.0 + 1.0) && ((z + "World") == "Hello World")`
    e, err := New(evalstr, nil)
    if err != nil {
        return
    }
    fmt.Printf("%v = %v", evalstr, e.Eval(map[string]interface{}{
		"x": 3.0,
		"y": 2.0,
		"z": "Hello ",
	}))
}
```

### 2.2、考虑性能，使用pool和缓存    

- 创建pool对象，如果有内置函数，则传入内置函数列表    
- 根据表达式，New一个对象，并传入pool对象   
- 如果考虑性能，可以开启缓存值（使用 `WithCacheValues(true)`）  

```go
var builtin = map[string]BuiltinFn{
	"sum": func(args ...interface{}) (interface{}, error) {
		var sum int64
		for _, v := range args {
			if v1, ok := v.(int64); !ok {
				return nil, fmt.Errorf("%v int64 invalid", v)
			} else {
				sum += v1
			}
		}
		return sum, nil
	},
}

func main() {
    const evalstr = `1+2*3+sum(1,2,3)`
	pool, _ := NewPool(WithBuiltinList(builtin))
    e, err := New(evalstr, pool, WithCacheValues(true))
    if err != nil {
        return
    }
    fmt.Printf("%v = %v", evalstr, e.Eval())
}
```

## 3、压测性能   

经过几个版本迭代的优化，增加缓存，减少拷贝，在性能方面，已经可以满足大部分场景。    

**测试环境：**   
- 测试机：MacBook Pro (13-inch, M1, 2020)   
- 测试代码：  
```go
const BenchmarkExpr = `((2 << 3) + (10 % 3)) * (5 - (x * 2)) + (3.0 / y) * (2.0 + 1.0) && ((z + "World") == "Hello World")`

func BenchmarkExprCache(b *testing.B) {
	pool, _ := NewPool(WithBuiltinList(builtin))
	e, err := New(BenchmarkExpr, pool, WithCacheValues(true))
	if err != nil {
		b.Errorf("expr New error = %v", err)
		return
	}
	for i := 0; i < b.N; i++ {
		_, err := e.Eval(map[string]interface{}{
			"x": 3.0,
			"y": 2.0,
			"z": "Hello ",
		})
		if err != nil {
			b.Errorf("expr Eval error = %v", err)
			return
		}
	}
}
``` 

**测试结果：**
```
% go test -bench . -benchtime 1s -cpu 4 -benchmem -cpuprofile cpu.pprof -memprofile mem.pprof
goos: darwin
goarch: arm64
pkg: github.com/linkxzhou/mylib/go/expr
BenchmarkExprCache-4     	 2193422	       543.7 ns/op	     192 B/op	      19 allocs/op
PASS
ok  	github.com/linkxzhou/mylib/go/expr	5.394s
```
**执行QPS：219w/s，平均每次执行耗时543.7纳秒，不过对于原生的性能，差别还是比较大，后续会将影响性能的递归改为循环方式，预计可以提升30%**     

## Benchmark
expr = `((2 << 3) + (10 % 3)) * (5 - (x * 2)) + (3.0 / y) * (2.0 + 1.0) && ((z + "World") == "Hello World")`
args = `"x": 3.0, "y": 2.0, "z": "Hello "`

```
% go test -bench . -benchtime 10s -cpu 1,2,4,8,16 -benchmem
goos: darwin
goarch: arm64
pkg: expr
cpu: Apple M4 Pro
BenchmarkExprNoCache            	27617198	       426.7 ns/op	     248 B/op	      24 allocs/op
BenchmarkExprNoCache-2          	29206125	       414.2 ns/op	     248 B/op	      24 allocs/op
BenchmarkExprNoCache-4          	28190004	       415.2 ns/op	     248 B/op	      24 allocs/op
BenchmarkExprNoCache-8          	28062582	       419.1 ns/op	     248 B/op	      24 allocs/op
BenchmarkExprNoCache-16         	28371337	       423.0 ns/op	     248 B/op	      24 allocs/op
BenchmarkExprCache              	35036452	       340.5 ns/op	     192 B/op	      19 allocs/op
BenchmarkExprCache-2            	37748946	       325.6 ns/op	     192 B/op	      19 allocs/op
BenchmarkExprCache-4            	37678604	       328.7 ns/op	     192 B/op	      19 allocs/op
BenchmarkExprCache-8            	35969542	       328.3 ns/op	     192 B/op	      19 allocs/op
BenchmarkExprCache-16           	35837248	       334.2 ns/op	     192 B/op	      19 allocs/op
BenchmarkBuiltinCall            	1000000000	         8.063 ns/op	       0 B/op	       0 allocs/op
BenchmarkBuiltinCall-2          	1000000000	         7.852 ns/op	       0 B/op	       0 allocs/op
BenchmarkBuiltinCall-4          	1000000000	         8.027 ns/op	       0 B/op	       0 allocs/op
BenchmarkBuiltinCall-8          	1000000000	         7.915 ns/op	       0 B/op	       0 allocs/op
BenchmarkBuiltinCall-16         	1000000000	         8.029 ns/op	       0 B/op	       0 allocs/op
BenchmarkDeepNested             	 5989566	      2016 ns/op	       0 B/op	       0 allocs/op
BenchmarkDeepNested-2           	 6028671	      2028 ns/op	       0 B/op	       0 allocs/op
BenchmarkDeepNested-4           	 5970871	      2027 ns/op	       0 B/op	       0 allocs/op
BenchmarkDeepNested-8           	 6098035	      1958 ns/op	       0 B/op	       0 allocs/op
BenchmarkDeepNested-16          	 6142263	      1974 ns/op	       0 B/op	       0 allocs/op
BenchmarkDeepNestedNative       	441792014	        27.59 ns/op	       0 B/op	       0 allocs/op
BenchmarkDeepNestedNative-2     	439887910	        27.64 ns/op	       0 B/op	       0 allocs/op
BenchmarkDeepNestedNative-4     	435334656	        28.03 ns/op	       0 B/op	       0 allocs/op
BenchmarkDeepNestedNative-8     	434936805	        27.87 ns/op	       0 B/op	       0 allocs/op
BenchmarkDeepNestedNative-16    	441028744	        28.16 ns/op	       0 B/op	       0 allocs/op
PASS
````