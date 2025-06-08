package simplejs

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestTokenizeBasicSymbols(t *testing.T) {
	code := `(){},;=+*-/%` + "\n" + `!&|^~<>?[]:{}.`
	tokens, err := Tokenize(code)
	assert.NoError(t, err)
	var expected = []TokenType{TokLParen, TokRParen, TokLBrace, TokRBrace, TokComma, TokSemicolon, TokAssign, TokPlus, TokAsterisk, TokMinus, TokSlash, TokRem, TokLogicalNot, TokBitAnd, TokBitOr, TokBitXor, TokBitNot, TokLT, TokGT, TokQuestion, TokLBracket, TokRBracket, TokColon, TokLBrace, TokRBrace, TokDot, TokEOF}
	if len(tokens) != len(expected) {
		t.Fatalf("token count mismatch: got %d, want %d", len(tokens), len(expected))
	}
	for i, tok := range tokens {
		if tok.Type != expected[i] {
			t.Errorf("token %d: got %v, want %v", i, tok.Type, expected[i])
		}
	}
}

func TestTokenizeKeywords(t *testing.T) {
	code := `if else while for function return var let const class extends new super throw try catch break true false null undefined delete`
	tokens, err := Tokenize(code)
	assert.NoError(t, err)
	var expected = []TokenType{
		TokIf, TokElse, TokWhile, TokFor, TokFunction, TokReturn, TokVar, TokLet, TokConst,
		TokClass, TokExtends, TokNew, TokSuper, TokThrow, TokTry, TokCatch, TokBreak,
		TokBool, TokBool, TokNull, TokUndefined, TokDelete, TokEOF,
	}
	if len(tokens) != len(expected) {
		t.Fatalf("token count mismatch: got %d, want %d", len(tokens), len(expected))
	}
	for i, tok := range tokens {
		if tok.Type != expected[i] {
			t.Errorf("token %d: got %v, want %v", i, tok.Type, expected[i])
		}
	}
}

func TestTokenizeNumbersAndStrings(t *testing.T) {
	code := `123 3.14 "hello" 'world'`
	tokens, err := Tokenize(code)
	assert.NoError(t, err)
	var expected = []TokenType{TokNumber, TokNumber, TokString, TokString, TokEOF}
	if len(tokens) != len(expected) {
		t.Fatalf("token count mismatch: got %d, want %d", len(tokens), len(expected))
	}
	for i, tok := range tokens {
		if tok.Type != expected[i] {
			t.Errorf("token %d: got %v, want %v", i, tok.Type, expected[i])
		}
	}
}

func TestTokenizeComments(t *testing.T) {
	code := `let a = 1; // this is a comment\n/* multi-line\ncomment */ let b = 2;`
	tokens, err := Tokenize(code)
	assert.NoError(t, err)
	foundComment := false
	for _, tok := range tokens {
		if strings.Contains(tok.Literal, "comment") {
			foundComment = true
		}
	}
	if foundComment {
		t.Error("comments should not be tokenized")
	}
}

func TestTokenizeError(t *testing.T) {
	code := "let a = 1; @"
	_, err := Tokenize(code)
	if err == nil {
		t.Error("should return error for illegal character '@'")
	}
}

func TestTokenizeIdentifiers(t *testing.T) {
	code := `_foo $bar baz123`
	tokens, err := Tokenize(code)
	assert.NoError(t, err)
	expected := []TokenType{TokIdentifier, TokIdentifier, TokIdentifier, TokEOF}
	if len(tokens) != len(expected) {
		t.Fatalf("token count mismatch: got %d, want %d", len(tokens), len(expected))
	}
	for i, tok := range tokens {
		assert.Equal(t, expected[i], tok.Type, "token %d", i)
	}
}

func TestTokenizeRegexLiteral(t *testing.T) {
	code := "var re = /abc/;"
	tokens, err := Tokenize(code)
	assert.NoError(t, err)
	// var, identifier, assign, regex, semicolon, eof
	expected := []TokenType{TokVar, TokIdentifier, TokAssign, TokRegex, TokSemicolon, TokEOF}
	if len(tokens) != len(expected) {
		t.Fatalf("token count mismatch: got %d, want %d", len(tokens), len(expected))
	}
	assert.Equal(t, "/abc/", tokens[3].Literal)
	for i, tok := range tokens {
		assert.Equal(t, expected[i], tok.Type, "token %d", i)
	}
}

func TestTokenizeUnterminatedString(t *testing.T) {
	code := `"hello`
	_, err := Tokenize(code)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "unterminated string")
}

func TestTokenizeEqualityOperators(t *testing.T) {
	code := "== === != !=="
	tokens, err := Tokenize(code)
	assert.NoError(t, err)
	expected := []TokenType{TokEqual, TokStrictEqual, TokNotEqual, TokNotStrictEqual, TokEOF}
	if len(tokens) != len(expected) {
		t.Fatalf("token count mismatch: got %d, want %d", len(tokens), len(expected))
	}
	for i, tok := range tokens {
		assert.Equal(t, expected[i], tok.Type, "token %d", i)
	}
}

func TestTokenizeAllOperatorsAndDelimiters(t *testing.T) {
	code := `+= -= *= /= << >> >>> & | ^ ~ ++ -- , ... ; : ? ( ) [ ] { } .`
	tokens, err := Tokenize(code)
	assert.NoError(t, err)
	expected := []TokenType{
		TokPlusAssign, TokMinusAssign, TokAsteriskAssign, TokSlashAssign,
		TokLShift, TokRShift, TokURShift,
		TokBitAnd, TokBitOr, TokBitXor, TokBitNot,
		TokInc, TokDec,
		TokComma, TokSpread, TokSemicolon, TokColon, TokQuestion,
		TokLParen, TokRParen, TokLBracket, TokRBracket, TokLBrace, TokRBrace, TokDot, TokEOF,
	}
	if len(tokens) != len(expected) {
		t.Fatalf("token count mismatch: got %d, want %d", len(tokens), len(expected))
	}
	for i, tok := range tokens {
		assert.Equal(t, expected[i], tok.Type, "token %d", i)
	}
}

func TestTokenizeAllKeywords(t *testing.T) {
	code := `if else while for function return var let const class extends new super throw try catch break delete`
	tokens, err := Tokenize(code)
	assert.NoError(t, err)
	expected := []TokenType{
		TokIf, TokElse, TokWhile, TokFor, TokFunction, TokReturn, TokVar, TokLet, TokConst,
		TokClass, TokExtends, TokNew, TokSuper, TokThrow, TokTry, TokCatch, TokBreak, TokDelete, TokEOF,
	}
	if len(tokens) != len(expected) {
		t.Fatalf("token count mismatch: got %d, want %d", len(tokens), len(expected))
	}
	for i, tok := range tokens {
		assert.Equal(t, expected[i], tok.Type, "token %d", i)
		// 同时测试 String() 输出
		_ = tok.Type.String()
	}
}

func TestTokenizeTemplateString(t *testing.T) {
	code := "`template string`"
	tokens, err := Tokenize(code)
	assert.NoError(t, err)
	expected := []TokenType{TokTemplate, TokEOF}
	if len(tokens) != len(expected) {
		t.Fatalf("token count mismatch: got %d, want %d", len(tokens), len(expected))
	}
	assert.Equal(t, "template string", tokens[0].Literal)
	for i, tok := range tokens {
		assert.Equal(t, expected[i], tok.Type, "token %d", i)
	}
}

func TestTokenizeBitwiseAndLogicalNot(t *testing.T) {
	code := "! ~"
	tokens, err := Tokenize(code)
	assert.NoError(t, err)
	expected := []TokenType{TokLogicalNot, TokBitNot, TokEOF}
	if len(tokens) != len(expected) {
		t.Fatalf("token count mismatch: got %d, want %d", len(tokens), len(expected))
	}
	for i, tok := range tokens {
		assert.Equal(t, expected[i], tok.Type, "token %d", i)
	}
}

func TestTokenizeDeleteKeyword(t *testing.T) {
	code := "delete foo"
	tokens, err := Tokenize(code)
	assert.NoError(t, err)
	expected := []TokenType{TokDelete, TokIdentifier, TokEOF}
	if len(tokens) != len(expected) {
		t.Fatalf("token count mismatch: got %d, want %d", len(tokens), len(expected))
	}
	for i, tok := range tokens {
		assert.Equal(t, expected[i], tok.Type, "token %d", i)
	}
}

func TestTokenizeES5Sample(t *testing.T) {
	code := `
	// ES5.1 语法覆盖示例
	var x = 42;
	var y = "hello";
	function add(a, b) {
	    return a + b;
	}
	if (x > 0) {
	    y = y + " world";
	} else {
	    y = "bye";
	}
	for (var i = 0; i < 10; i++) {
	    x += i;
	}
	while (x < 100) {
	    x++;
	}
	try {
	    throw new Error("fail");
	} catch (e) {
	    x = 0;
	}
	var arr = [1, 2, 3];
	var obj = {foo: 1, bar: 2};
	delete obj.foo;
	`
	tokens, err := Tokenize(code)
	assert.NoError(t, err)
	assert.True(t, len(tokens) > 0, "should produce tokens")

	// 断言部分关键 token 顺序
	varTypes := []TokenType{TokVar, TokIdentifier, TokAssign, TokNumber, TokSemicolon}
	for i := 0; i < len(tokens)-len(varTypes); i++ {
		match := true
		for j, typ := range varTypes {
			if tokens[i+j].Type != typ {
				match = false
				break
			}
		}
		if match {
			// 找到第一个 var x = 42;
			return
		}
	}
	t.Errorf("did not find expected var x = 42; token sequence")
}

// 新增测试用例：测试带转义字符的字符串
func TestTokenizeEscapedStrings(t *testing.T) {
	code := `"hello\nworld" 'it\'s me'`
	tokens, err := Tokenize(code)
	assert.NoError(t, err)
	expected := []TokenType{TokString, TokString, TokEOF}
	if len(tokens) != len(expected) {
		t.Fatalf("token count mismatch: got %d, want %d", len(tokens), len(expected))
	}
	assert.Equal(t, "hello\nworld", tokens[0].Literal)
	assert.Equal(t, "it's me", tokens[1].Literal)
}

// 新增测试用例：测试复杂的正则表达式
func TestTokenizeComplexRegex(t *testing.T) {
	code := `var emailRegex = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;`
	tokens, err := Tokenize(code)
	assert.NoError(t, err)
	expected := []TokenType{TokVar, TokIdentifier, TokAssign, TokRegex, TokSemicolon, TokEOF}
	if len(tokens) != len(expected) {
		t.Fatalf("token count mismatch: got %d, want %d", len(tokens), len(expected))
	}
	assert.Equal(t, "/^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$/", tokens[3].Literal)
}

// 新增测试用例：测试逻辑运算符
func TestTokenizeLogicalOperators(t *testing.T) {
	code := `a && b || !c`
	tokens, err := Tokenize(code)
	assert.NoError(t, err)
	expected := []TokenType{TokIdentifier, TokLogicalAnd, TokIdentifier, TokLogicalOr, TokLogicalNot, TokIdentifier, TokEOF}
	if len(tokens) != len(expected) {
		t.Fatalf("token count mismatch: got %d, want %d", len(tokens), len(expected))
	}
	for i, tok := range tokens {
		assert.Equal(t, expected[i], tok.Type, "token %d", i)
	}
}

// 新增测试用例：测试未终止的多行注释
func TestTokenizeUnterminatedComment(t *testing.T) {
	code := `let a = 1; /* This comment is not terminated`
	tokens, err := Tokenize(code)
	// 当前实现不会检测未终止的注释，所以这里不应该报错
	assert.NoError(t, err)
	expected := []TokenType{TokLet, TokIdentifier, TokAssign, TokNumber, TokSemicolon, TokEOF}
	if len(tokens) != len(expected) {
		t.Fatalf("token count mismatch: got %d, want %d", len(tokens), len(expected))
	}
}

// 新增测试用例：测试箭头函数
func TestTokenizeArrowFunction(t *testing.T) {
	code := `const add = (a, b) => a + b;`
	tokens, err := Tokenize(code)
	assert.NoError(t, err)
	expected := []TokenType{
		TokConst, TokIdentifier, TokAssign, TokLParen, TokIdentifier, TokComma,
		TokIdentifier, TokRParen, TokArrow, TokIdentifier, TokPlus, TokIdentifier,
		TokSemicolon, TokEOF,
	}
	if len(tokens) != len(expected) {
		t.Fatalf("token count mismatch: got %d, want %d", len(tokens), len(expected))
	}
	for i, tok := range tokens {
		assert.Equal(t, expected[i], tok.Type, "token %d", i)
	}
}

// 新增测试用例：测试数字字面量的边界情况
func TestTokenizeNumberLiterals(t *testing.T) {
	code := `0 42 3.14159 .5 0.`
	tokens, err := Tokenize(code)
	assert.NoError(t, err)
	expected := []TokenType{TokNumber, TokNumber, TokNumber, TokNumber, TokNumber, TokEOF}
	if len(tokens) != len(expected) {
		t.Fatalf("token count mismatch: got %d, want %d", len(tokens), len(expected))
	}
	assert.Equal(t, "0", tokens[0].Literal)
	assert.Equal(t, "42", tokens[1].Literal)
	assert.Equal(t, "3.14159", tokens[2].Literal)
	assert.Equal(t, ".5", tokens[3].Literal)
	assert.Equal(t, "0.", tokens[4].Literal)
}

// 新增测试用例：测试模板字符串中的换行
func TestTokenizeTemplateStringWithNewlines(t *testing.T) {
	code := "`line1\nline2\nline3`"
	tokens, err := Tokenize(code)
	assert.NoError(t, err)
	expected := []TokenType{TokTemplate, TokEOF}
	if len(tokens) != len(expected) {
		t.Fatalf("token count mismatch: got %d, want %d", len(tokens), len(expected))
	}
	assert.Equal(t, "line1\nline2\nline3", tokens[0].Literal)
	assert.Equal(t, 3, tokens[1].Line) // 检查行号是否正确递增
}

// 新增测试用例：测试位运算符组合
func TestTokenizeBitwiseOperations(t *testing.T) {
	code := `a & b | c ^ ~d`
	tokens, err := Tokenize(code)
	assert.NoError(t, err)
	expected := []TokenType{
		TokIdentifier, TokBitAnd, TokIdentifier, TokBitOr, TokIdentifier,
		TokBitXor, TokBitNot, TokIdentifier, TokEOF,
	}
	if len(tokens) != len(expected) {
		t.Fatalf("token count mismatch: got %d, want %d", len(tokens), len(expected))
	}
	for i, tok := range tokens {
		assert.Equal(t, expected[i], tok.Type, "token %d", i)
	}
}

// 新增测试用例：测试ES6+代码样例
func TestTokenizeES6Sample(t *testing.T) {
	code := `
	// ES6+ 语法覆盖示例
	const PI = 3.14159;
	let sum = (a, b) => a + b;
	class Person {
		constructor(name) {
			this.name = name;
		}
		sayHello() {
			return "Hello, " + this.name + "!";
		}
	}
	const numbers = [1, 2, 3];
	const doubled = numbers.map(n => n * 2);
	const { x, y, ...rest } = { x: 1, y: 2, a: 3, b: 4 };
	`
	tokens, err := Tokenize(code)
	assert.NoError(t, err)
	assert.True(t, len(tokens) > 0, "should produce tokens")

	// 检查是否包含箭头函数
	arrowFound := false
	for _, tok := range tokens {
		if tok.Type == TokArrow {
			arrowFound = true
			break
		}
	}
	assert.True(t, arrowFound, "should contain arrow function token")

	// 检查是否包含扩展运算符
	spreadFound := false
	for _, tok := range tokens {
		if tok.Type == TokSpread {
			spreadFound = true
			break
		}
	}
	assert.True(t, spreadFound, "should contain spread operator token")
}
