package simplejs

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestParseFunctionDeclStatement(t *testing.T) {
	code := "function foo(a, b) { return a + b; }"
	toks, err := Tokenize(code)
	assert.NoError(t, err)
	ctx := &RunContext{global: NewScope(nil)}
	p := NewParser(toks, ctx)
	stmt, err := p.ParseStatement()
	assert.NoError(t, err)
	exprStmt, ok := stmt.(*ExpressionStmt)
	if !ok {
		t.Fatalf("stmt is not *ExpressionStmt, got %T", stmt)
	}
	fn, ok := exprStmt.Expr.(*FunctionDecl)
	if !ok {
		t.Fatalf("exprStmt.Expr is not *FunctionDecl, got %T", exprStmt.Expr)
	}
	assert.Equal(t, "foo", fn.Name.Name)
	assert.Len(t, fn.Params, 2)
}

func TestParseFunctionExpr(t *testing.T) {
	code := "function(a) { }"
	toks, err := Tokenize(code)
	assert.NoError(t, err)
	ctx := &RunContext{global: NewScope(nil)}
	p := NewParser(toks, ctx)
	stmt, err := p.ParseStatement()
	assert.NoError(t, err)
	exprStmt, ok := stmt.(*ExpressionStmt)
	if !ok {
		t.Fatalf("stmt is not *ExpressionStmt, got %T", stmt)
	}
	fn, ok := exprStmt.Expr.(*FunctionDecl)
	if !ok {
		t.Fatalf("exprStmt.Expr is not *FunctionDecl, got %T", exprStmt.Expr)
	}
	assert.Nil(t, fn.Name)
	assert.Len(t, fn.Params, 1)
}

func TestParseNewExpr(t *testing.T) {
	code := "new Obj(1, 2)"
	toks, err := Tokenize(code)
	assert.NoError(t, err)
	ctx := &RunContext{global: NewScope(nil)}
	p := NewParser(toks, ctx)
	expr, err := p.ParseExpression()
	assert.NoError(t, err)
	ne, ok := expr.(*NewExpr)
	if !ok {
		t.Fatalf("expr is not *NewExpr, got %T", expr)
	}
	ident, ok := ne.Callee.(*Identifier)
	if !ok {
		t.Fatalf("ne.Callee is not *Identifier, got %T", ne.Callee)
	}
	assert.Equal(t, "Obj", ident.Name)
	assert.Len(t, ne.Arguments, 2)
}

func TestParseFunctionAddStatement(t *testing.T) {
	code := `function add(a, b) { return a + b; }
let r = add(2, 3);
print("r = ", r);
if (r !== 5) throw 'fail: r !== 5';`
	toks, err := Tokenize(code)
	assert.NoError(t, err)
	ctx := &RunContext{global: NewScope(nil)}
	p := NewParser(toks, ctx)
	prog, err := p.ParseProgram()
	assert.NoError(t, err)
	assert.NotNil(t, prog)
	// 检查AST结构
	stmts := prog.Body
	assert.GreaterOrEqual(t, len(stmts), 2)
	// 检查第一个语句是函数声明
	var fnDecl *FunctionDecl
	switch node := stmts[0].(type) {
	case *FunctionDecl:
		fnDecl = node
	case *ExpressionStmt:
		var ok bool
		fnDecl, ok = node.Expr.(*FunctionDecl)
		if !ok {
			t.Fatalf("node.Expr is not *FunctionDecl, got %T", node.Expr)
		}
	default:
		t.Fatalf("unexpected type for first statement: %T", node)
	}
	assert.NotNil(t, fnDecl)
	assert.Equal(t, "add", fnDecl.Name.Name)
	assert.Len(t, fnDecl.Params, 2)
	// 检查第二个语句是let声明
	varDecl, ok := stmts[1].(*VarDecl)
	if !ok {
		t.Fatalf("stmts[1] is not *VarDecl, got %T", stmts[1])
	}
	id, ok := varDecl.Name.(*Identifier)
	if !ok {
		t.Fatalf("varDecl.Name is not *Identifier, got %T", varDecl.Name)
	}
	assert.Equal(t, "r", id.Name)
}

func TestParseClosureStatement(t *testing.T) {
	code := `
function makeAdder(x) {
  return function(y) { return x + y; };
}
let add2 = makeAdder(2);
let r = add2(3);
print("r =", r);
if (r !== 5) throw 'fail: closure failed';
`
	toks, err := Tokenize(code)
	assert.NoError(t, err)
	ctx := &RunContext{global: NewScope(nil)}
	p := NewParser(toks, ctx)
	prog, err := p.ParseProgram()
	assert.NoError(t, err)
	assert.NotNil(t, prog)
	stmts := prog.Body
	assert.GreaterOrEqual(t, len(stmts), 3)
	// 检查第一个语句是函数声明 makeAdder
	var fnDecl *FunctionDecl
	switch node := stmts[0].(type) {
	case *FunctionDecl:
		fnDecl = node
	case *ExpressionStmt:
		var ok bool
		fnDecl, ok = node.Expr.(*FunctionDecl)
		if !ok {
			t.Fatalf("node.Expr is not *FunctionDecl, got %T", node.Expr)
		}
	default:
		t.Fatalf("unexpected type for first statement: %T", node)
	}
	assert.NotNil(t, fnDecl)
	assert.Equal(t, "makeAdder", fnDecl.Name.Name)
	assert.Len(t, fnDecl.Params, 1)
	// 检查 makeAdder 返回值是匿名函数
	retStmt, ok := fnDecl.Body.Body[0].(*ReturnStmt)
	if !ok {
		t.Fatalf("fnDecl.Body.Body[0] is not *ReturnStmt, got %T", fnDecl.Body.Body[0])
	}
	innerFn, ok := retStmt.Argument.(*FunctionDecl)
	if !ok {
		t.Fatalf("retStmt.Argument is not *FunctionDecl, got %T", retStmt.Argument)
	}
	assert.Nil(t, innerFn.Name)
	assert.Len(t, innerFn.Params, 1)
	// 检查 let add2 = makeAdder(2)
	varDecl, ok := stmts[1].(*VarDecl)
	if !ok {
		t.Fatalf("stmts[1] is not *VarDecl, got %T", stmts[1])
	}
	id, ok := varDecl.Name.(*Identifier)
	if !ok {
		t.Fatalf("varDecl.Name is not *Identifier, got %T", varDecl.Name)
	}
	assert.Equal(t, "add2", id.Name)
}

func TestParseArrowFunction(t *testing.T) {
	code := "(x) => x + 1"
	toks, err := Tokenize(code)
	assert.NoError(t, err)
	ctx := &RunContext{global: NewScope(nil)}
	p := NewParser(toks, ctx)
	expr, err := p.ParseExpression()
	assert.NoError(t, err)
	arrowFn, ok := expr.(*ArrowFunctionExpr)
	if !ok {
		t.Fatalf("expr is not *ArrowFunctionExpr, got %T", expr)
	}
	assert.Len(t, arrowFn.Params, 1)
}

// === parser_functions.go 函数调用与 this/原型链/箭头函数/原生函数测试 ===
func TestFunctionCallVariants(t *testing.T) {
	cases := []struct {
		code    string
		expect  string
		desc    string
		wantErr bool
	}{
		// 普通函数调用
		{"function f(x) { return x+1; } f(2);", "3", "global function call", false},
		// // 闭包
		// {"function makeAdder(a) { return function(b) { return a+b; }; } let add5 = makeAdder(5); add5(3);", "8", "closure function call", false},
		// // 对象方法调用 this 绑定
		// {"let obj = {x: 42, get: function() { return this.x; }}; obj.get();", "42", "object method call with this", false},
		// // 原型链方法
		// {"let proto = {foo: function() { return 99; }}; let obj = {}; obj.__proto__ = proto; obj.foo();", "99", "method via prototype chain", false},
		// // 箭头函数 this 继承
		// {"let that = 7; let f = () => this; f.call({val:that});", "[object Object]", "arrow function this inheritance (should not be bound)", false},
		// // Go 注册原生函数 print 返回 undefined
		// {"typeof print('this is print function');", "undefined", "native Go function call", false},
		// // 错误: 非函数调用
		// {"let a = 1; a();", "", "call non-function should error", true},
		// // 错误: 调用不存在的方法
		// {"let obj = {}; obj.noSuchMethod();", "", "call missing method should error", true},
	}
	for _, c := range cases {
		t.Logf("Testing code: %s, desc: %s", c.code, c.desc)
		if c.wantErr {
			_, err := runJSWithError(t, c.code)
			if err == nil {
				t.Errorf("%s: expect error, got nil", c.desc)
			}
		} else {
			result := runJS(t, c.code)
			if result.ToString() != c.expect {
				t.Errorf("%s: expect %v, got %v", c.desc, c.expect, result.ToString())
			}
		}
	}
}
