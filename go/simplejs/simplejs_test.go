package simplejs

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

func registerPrint(ctx *RunContext) {
	ctx.RegisterGoFunc("print", func(args ...JSValue) JSValue {
		for i, v := range args {
			if i > 0 {
				fmt.Print(" ")
			}
			fmt.Print(v.ToString())
		}
		fmt.Println()
		return Undefined()
	})
}

func runJS(t *testing.T, code string) JSValue {
	ctx := &RunContext{global: NewScope(nil)}
	registerPrint(ctx)
	tokens, err := Tokenize(code)
	assert.NoError(t, err)
	parser := NewParser(tokens, ctx)
	val, err := parser.ParseProgram()
	if err != nil {
		t.Logf("DebugInfo: \n%v", strings.Join(parser.debugString, "\n"))
	}
	assert.NoError(t, err)
	return val
}

func runJSWithError(t *testing.T, code string) (JSValue, error) {
	ctx := NewContext(1024)
	registerPrint(ctx)
	lines := splitLines(code)
	ctx.sourceLines = &lines
	tokens, err := Tokenize(code)
	if err != nil {
		return Undefined(), err
	}
	parser := NewParser(tokens, ctx)
	val, err := parser.ParseProgram()
	if err != nil {
		t.Logf("DebugInfo: \n%v", strings.Join(parser.debugString, "\n"))
	}
	return val, err
}

func TestLetConstBlockScope(t *testing.T) {
	code := `
let x = 1;
{
  let x = 2;
  const y = 3;
  if (x !== 2) throw 'fail: x !== 2';
}
if (x !== 1) throw 'fail: x !== 1';
x
`
	result := runJS(t, code)
	if result.Type != JSNumber || result.Number != 1 {
		t.Errorf("Expected x to be 1, got %v (type: %v)", result.ToString(), result.Type.String())
	}
}

func TestClassInheritance(t *testing.T) {
	code := `
class Animal {
  constructor(name) {
    this.name = name;
  }
  speak() {
    return this.name + " makes a noise.";
  }
}

class Dog extends Animal {
  constructor(name) {
    super(name);
  }
  speak() {
    return this.name + " barks.";
  }
}

let dog = new Dog("Rex");
if (dog.speak() !== "Rex barks.") throw 'fail: dog.speak() !== "Rex barks."';
dog.speak()
`
	result := runJS(t, code)
	if result.Type != JSString || result.String != "Rex barks." {
		t.Errorf("Expected 'Rex barks.', got %v (type: %v)", result.ToString(), result.Type.String())
	}
}

func TestObjectMethodCallThis(t *testing.T) {
	code := `
let obj = {
  x: 10,
  getX: function() {
    return this.x;
  }
};
if (obj.getX() !== 10) throw 'fail: obj.getX() !== 10';
obj.getX()
`
	result := runJS(t, code)
	if result.Type != JSNumber || result.Number != 10 {
		t.Errorf("Expected 10, got %v (type: %v)", result.ToString(), result.Type.String())
	}
}

func TestDestructuringAssignment(t *testing.T) {
	code := `
let obj = {a: 1, b: 2};
let {a, b} = obj;
if (a !== 1 || b !== 2) throw 'fail: a !== 1 || b !== 2';
let arr = [3,4];
let [x, y] = arr;
if (x !== 3 || y !== 4) throw 'fail: x !== 3 || y !== 4';
[y, x]
`
	result := runJS(t, code)
	if !result.IsArrayType() || len(result.ToObject())-1 != 2 {
		t.Errorf("Expected array with 2 elements, got %v (type: %v)", result.ToString(), result.Type.String())
	}

	// 验证数组内容
	arr := result.ToObject()
	if arr["0"].ToNumber() != 4 || arr["1"].ToNumber() != 3 {
		t.Errorf("Expected [4, 3], got %v", result.ToString())
	}
}

// 变量声明与赋值
func TestVarAssignment(t *testing.T) {
	code := `
let a = 5;
if (a !== 5) throw 'fail: a !== 5';
a = 7;
if (a !== 7) throw 'fail: a !== 7';
a
`
	result := runJS(t, code)
	if result.Type != JSNumber || result.Number != 7 {
		t.Errorf("Expected 7, got %v (type: %v)", result.ToString(), result.Type.String())
	}
}

// 算术运算
func TestArithmetic(t *testing.T) {
	code := `
let a = 1 + 2 * 3 - 4 / 2;
if (a !== 5) throw 'fail: a !== 5';
a
`
	result := runJS(t, code)
	if result.Type != JSNumber || result.Number != 5 {
		t.Errorf("Expected 5, got %v (type: %v)", result.ToString(), result.Type.String())
	}
}

// 布尔逻辑
func TestBooleanLogic(t *testing.T) {
	code := `
let b = true && false || !false;
if (b !== true) throw 'fail: b !== true';
b
`
	result := runJS(t, code)
	if result.Type != JSBoolean || !result.Bool {
		t.Errorf("Expected true, got %v (type: %v)", result.ToString(), result.Type.String())
	}
}

// 字符串操作
func TestStringOps(t *testing.T) {
	code := `
let s = "foo" + "bar";
if (s !== "foobar") throw 'fail: s !== "foobar"';
s
`
	result := runJS(t, code)
	if result.Type != JSString || result.String != "foobar" {
		t.Errorf("Expected 'foobar', got %v (type: %v)", result.ToString(), result.Type.String())
	}
}

// 数组基本操作
func TestArrayOps(t *testing.T) {
	code := `
let arr = [1,2,3];
if (arr[0] !== 1 || arr[2] !== 3) throw 'fail: arr[0] !== 1 || arr[2] !== 3';
arr
`
	result := runJS(t, code)
	if result.Type != JSObject {
		t.Errorf("Expected array object, got %v (type: %v)", result.ToString(), result.Type.String())
	}

	// 验证数组内容
	arr := result.ToObject()
	if arr["length"].ToNumber() != 3 || arr["0"].ToNumber() != 1 || arr["1"].ToNumber() != 2 || arr["2"].ToNumber() != 3 {
		t.Errorf("Expected [1,2,3], got %v", result.ToString())
	}
}

// 对象属性访问
func TestObjectProps(t *testing.T) {
	code := `
let obj = {x: 10, y: 20};
if (obj.x !== 10 || obj.y !== 20) throw 'fail: obj.x !== 10 || obj.y !== 20';
obj
`
	result := runJS(t, code)
	if result.Type != JSObject {
		t.Errorf("Expected object, got %v (type: %v)", result.ToString(), result.Type.String())
	}

	// 验证对象属性
	obj := result.ToObject()
	if len(obj) != 2 || obj["x"].ToNumber() != 10 || obj["y"].ToNumber() != 20 {
		t.Errorf("Expected {x:10, y:20}, got %v", result.ToString())
	}
}

// 简单函数调用与返回
func TestFunctionCall(t *testing.T) {
	code := `
function add(a, b) {
  return a + b;
}
let r = add(2, 3);
if (r !== 5) throw 'fail: r !== 5';
r
`
	result := runJS(t, code)
	if result.Type != JSNumber || result.Number != 5 {
		t.Errorf("Expected 5, got %v (type: %v)", result.ToString(), result.Type.String())
	}
}

// if/else 分支
func TestIfElse(t *testing.T) {
	code := `
let x = 2;
let y;
if (x > 1) {
  y = 10;
} else {
  y = 20;
}
if (y !== 10) throw 'fail: y !== 10';
y
`
	result := runJS(t, code)
	if result.Type != JSNumber || result.Number != 10 {
		t.Errorf("Expected 10, got %v (type: %v)", result.ToString(), result.Type.String())
	}
}

// while 循环
func TestWhileLoop(t *testing.T) {
	code := `
let i = 0;
let sum = 0;
while (i < 5) {
  sum = sum + i;
  i = i + 1;
}
if (sum !== 10) throw 'fail: sum !== 10';
sum
`
	result := runJS(t, code)
	if result.Type != JSNumber || result.Number != 10 {
		t.Errorf("Expected 10, got %v (type: %v)", result.ToString(), result.Type.String())
	}
}

// for 循环 (带 break)
func TestForLoopWithBreak(t *testing.T) {
	code := `
let sum = 0;
for (let i = 0; i < 10; i = i + 1) {
  if (i === 5) break;
  sum = sum + i;
}
if (sum !== 10) throw 'fail: sum !== 10';
sum
`
	result := runJS(t, code)
	if result.Type != JSNumber || result.Number != 10 {
		t.Errorf("Expected 10, got %v (type: %v)", result.ToString(), result.Type.String())
	}
}

// 嵌套作用域和闭包
func TestClosureScope(t *testing.T) {
	code := `
function makeAdder(x) {
  return function(y) { return x + y; };
}
let add5 = makeAdder(5);
if (add5(3) !== 8) throw 'fail';
add5(3)
`
	result := runJS(t, code)
	if result.Type != JSNumber || result.Number != 8 {
		t.Errorf("Expected 8, got %v (type: %v)", result.ToString(), result.Type.String())
	}
}

// 数组 push pop
func TestArrayPushPop(t *testing.T) {
	code := `
let arr = [1,2,3];
arr[3] = 4;
if (arr[3] !== 4) throw 'fail: arr[3] !== 4';
arr[4] = arr[3] + 1;
if (arr[4] !== 5) throw 'fail: arr[4] !== 5';
arr
`
	result := runJS(t, code)
	if result.Type != JSObject {
		t.Errorf("Expected array object, got %v (type: %v)", result.ToString(), result.Type.String())
	}

	// 验证数组内容
	arr := result.ToObject()
	if arr["length"].ToNumber() != 5 || arr["0"].ToNumber() != 1 || arr["4"].ToNumber() != 5 {
		t.Errorf("Expected array with 5 elements, got %v", result.ToString())
	}
}

// 对象属性删除
func TestObjectDelete(t *testing.T) {
	code := `
let obj = {a: 1, b: 2};
delete obj.a;
if (obj.a !== undefined) throw 'fail: obj.a !== undefined';
obj
`
	result := runJS(t, code)
	if result.Type != JSObject {
		t.Errorf("Expected object, got %v (type: %v)", result.ToString(), result.Type.String())
	}

	// 验证对象属性
	obj := result.ToObject()
	if len(obj) != 1 || obj["b"].ToNumber() != 2 || obj["a"].Type != JSUndefined {
		t.Errorf("Expected {b:2}, got %v", result.ToString())
	}
}

// 三元表达式
func TestTernaryOperator(t *testing.T) {
	code := `
let a = 5;
let b = a > 3 ? 10 : 20;
if (b !== 10) throw 'fail: b !== 10';
b
`
	result := runJS(t, code)
	if result.Type != JSNumber || result.Number != 10 {
		t.Errorf("Expected 10, got %v (type: %v)", result.ToString(), result.Type.String())
	}
}

// 字符串模板拼接
func TestStringTemplate(t *testing.T) {
	code := `
let name = "world";
let s = "Hello, " + name + "!";
if (s !== "Hello, world!") throw 'fail: s !== "Hello, world!"';
s
`
	result := runJS(t, code)
	if result.Type != JSString || result.String != "Hello, world!" {
		t.Errorf("Expected 'Hello, world!', got %v (type: %v)", result.ToString(), result.Type.String())
	}
}

// null/undefined 判断
func TestNullUndefined(t *testing.T) {
	code := `
let a = null;
let b;
if (a !== null) throw 'fail: a !== null';
if (b !== undefined) throw 'fail: b !== undefined';
[a, b]
`
	result := runJS(t, code)
	if result.Type != JSObject {
		t.Errorf("Expected array object, got %v (type: %v)", result.ToString(), result.Type.String())
	}

	// 验证数组内容
	arr := result.ToObject()
	if arr["length"].ToNumber() != 2 || arr["0"].Type != JSNull || arr["1"].Type != JSUndefined {
		t.Errorf("Expected [null, undefined], got %v", result.ToString())
	}
}

// super 关键字调用
func TestSuperCall(t *testing.T) {
	code := `
class A {
  foo() { return 1; }
}
class B extends A {
  foo() { return super.foo() + 2; }
}
let b = new B();
if (b.foo() !== 3) throw 'fail: b.foo() !== 3';
b.foo()
`
	result := runJS(t, code)
	if result.Type != JSNumber || result.Number != 3 {
		t.Errorf("Expected 3, got %v (type: %v)", result.ToString(), result.Type.String())
	}
}

// 箭头函数作用域
func TestArrowFunctionScope(t *testing.T) {
	code := `
let x = 10;
let f = () => x + 5;
if (f() !== 15) throw 'fail: f() !== 15';
f()
`
	result := runJS(t, code)
	if result.Type != JSNumber || result.Number != 15 {
		t.Errorf("Expected 15, got %v (type: %v)", result.ToString(), result.Type.String())
	}
}

// 测试后缀递增/递减操作符
func TestIncrementDecrement(t *testing.T) {
	code := `
let i = 5;
let j = i + 1;  // j = 5, i = 6
let k = i - 1;  // k = 5, i = 5
if (j !== 6 || k !== 4 || i !== 5) throw 'fail';

// 测试在表达式中使用
let a = 1;
let b = 2;
let c = a + b;
if (a !== 1 || b !== 2 || c !== 3) throw 'fail';

// 测试在循环中使用
let sum = 0;
for (let x = 0; x < 5; x = x + 1) {
  sum = sum + x;
  print("sum: ", sum, ", x: ", x)
}
if (sum !== 10) throw 'fail: sum !== 10';  // 0+1+2+3+4=10
[i, j, k, a, b, c, sum]
`
	result := runJS(t, code)
	if result.Type != JSObject {
		t.Errorf("Expected array object, got %v (type: %v)", result.ToString(), result.Type.String())
	}

	// 验证数组内容
	arr := result.ToObject()
	if arr["length"].ToNumber() != 7 {
		t.Errorf("Expected array with 7 elements, got %v", result.ToString())
	}

	// 验证各个值
	expectedValues := map[string]float64{
		"0": 5,  // i
		"1": 6,  // j
		"2": 4,  // k
		"3": 1,  // a
		"4": 2,  // b
		"5": 3,  // c
		"6": 10, // sum
	}

	for key, expected := range expectedValues {
		if arr[key].ToNumber() != expected {
			t.Errorf("Expected %s to be %v, got %v", key, expected, arr[key].ToNumber())
		}
	}
}

// 测试 RegisterGoFunc 注册 Go 函数到 JS
func TestRegisterGoFunc(t *testing.T) {
	ctx := NewContext(1024)
	// 注册一个简单的加法函数
	ctx.RegisterGoFunc("add", func(args ...JSValue) JSValue {
		if len(args) != 2 {
			return JSValue{Type: JSError, String: "add requires 2 arguments"}
		}
		a := args[0].ToNumber()
		b := args[1].ToNumber()
		return JSValue{Type: JSNumber, Number: a + b}
	})

	code := `
let result = add(3, 5);
if (result !== 8) throw 'fail: add(3,5) !== 8';
result;
`
	lines := splitLines(code)
	ctx.sourceLines = &lines
	tokens, err := Tokenize(code)
	assert.NoError(t, err)
	parser := NewParser(tokens, ctx)
	val, err := parser.ParseProgram()
	if err != nil {
		t.Logf("DebugInfo: \n%v", strings.Join(parser.debugString, "\n"))
	}
	assert.NoError(t, err)
	if val.Type != JSNumber || val.Number != 8 {
		t.Errorf("Expected 8, got %v (type: %v)", val.ToString(), val.Type.String())
	}
}

// 测试 throw 语句
func TestThrowStatement(t *testing.T) {
	// 测试简单的 throw 语句
	code0 := `
// 直接抛出异常并捕获
try {
  throw "Test exception";
} catch (e) {
  // 正确捕获异常
}

// 成功完成
"success"
`
	result := runJS(t, code0)
	if result.Type != JSString || result.String != "success" {
		t.Errorf("Expected 'success', got %v (type: %v)", result.ToString(), result.Type.String())
	}

	// 测试未捕获的 throw
	code2 := `
throw "Uncaught test exception";
`
	_, err := runJSWithError(t, code2)
	if err == nil {
		t.Errorf("Expected error but got none")
	} else if !strings.Contains(err.Error(), "Uncaught exception: Uncaught test exception") {
		t.Errorf("Expected 'Uncaught exception' but got: %v", err)
	}
}

// 测试 try-catch 语句
func TestTryCatch(t *testing.T) {
	code := `
// 简单的 try-catch
try {
  let x = 5;
} catch (e) {
  // 不应该执行到这里
}
"success";
`
	result := runJS(t, code)
	if result.Type != JSString || result.String != "success" {
		t.Errorf("Expected 'success', got %v (type: %v)", result.ToString(), result.Type.String())
	}
}

// 测试 !== 操作符
func TestNotStrictEqual(t *testing.T) {
	// 测试数字和字符串比较
	{
		code := `5 !== "5";`
		ctx := NewContext(1024)
		tokens, err := Tokenize(code)
		if err != nil {
			t.Fatalf("Tokenize error: %v", err)
		}

		parser := NewParser(tokens, ctx)
		result, err := parser.ParseProgram()
		if err != nil {
			t.Fatalf("Parse error: %v", err)
		}

		if !result.ToBool() {
			t.Errorf("5 !== \"5\" should be true, got %v (type: %v)", result.ToString(), result.Type.String())
		}

		// 验证结果类型
		if result.Type != JSBoolean {
			t.Errorf("Expected result to be boolean type, got %v", result.Type.String())
		}
	}

	// 测试数字和数字比较
	{
		code := `5 !== 5;`
		ctx := NewContext(1024)
		tokens, err := Tokenize(code)
		if err != nil {
			t.Fatalf("Tokenize error: %v", err)
		}

		parser := NewParser(tokens, ctx)
		result, err := parser.ParseProgram()
		if err != nil {
			t.Fatalf("Parse error: %v", err)
		}

		if result.ToBool() {
			t.Errorf("5 !== 5 should be false, got %v (type: %v)", result.ToString(), result.Type.String())
		}

		// 验证结果类型
		if result.Type != JSBoolean {
			t.Errorf("Expected result to be boolean type, got %v", result.Type.String())
		}
	}
}

// Debug test for currentLineInfo error line reporting
func TestCurrentLineInfo_Debug(t *testing.T) {
	ctx := NewContext(1024)
	code := `let a = 1;
let b = 2;
let c = ; // syntax error here
let d = 4;`
	// 设置源码行，这样 currentLineInfo 才能返回正确的代码行
	lines := splitLines(code)
	ctx.sourceLines = &lines
	tokens, err := Tokenize(code)
	if err != nil {
		t.Fatalf("tokenize error: %v", err)
	}
	parser := NewParser(tokens, ctx)
	_, err = parser.ParseProgram()
	if err == nil {
		t.Fatalf("expected parse error, got nil")
	}
	parseErr, ok := err.(ParseException)
	if !ok {
		t.Fatalf("expected ParseException, got %T: %v", err, err)
	}
	if parseErr.Line == 0 || parseErr.Code == "" {
		t.Errorf("currentLineInfo did not return correct line/code: got line=%d, code='%s'", parseErr.Line, parseErr.Code)
	}
	if parseErr.Line != 3 {
		t.Errorf("expected error at line 3, got %d", parseErr.Line)
	}
	if parseErr.Code != "let c = ; // syntax error here" {
		t.Errorf("expected code line 'let c = ; // syntax error here', got '%s'", parseErr.Code)
	}
}

// 遍历 examples 目录，逐个运行 js 文件
func TestExamplesFolder(t *testing.T) {
	dir := "./examples"
	files, err := os.ReadDir(dir)
	assert.NoError(t, err)
	for _, file := range files {
		if file.IsDir() || filepath.Ext(file.Name()) != ".js" {
			continue
		}
		jsPath := filepath.Join(dir, file.Name())
		content, err := os.ReadFile(jsPath)
		assert.NoError(t, err, "reading %s", file.Name())
		t.Run(file.Name(), func(t *testing.T) {
			ctx := &RunContext{global: NewScope(nil)}
			registerPrint(ctx)
			// 设置源码行，这样 currentLineInfo 才能返回正确的代码行
			lines := splitLines(string(content))
			ctx.sourceLines = &lines
			tokens, err := Tokenize(string(content))
			assert.NoError(t, err, "tokenize %s", file.Name())
			parser := NewParser(tokens, ctx)
			_, err = parser.ParseProgram()
			assert.NoError(t, err, "parse %s", file.Name())
		})
	}
}
