package simplejs

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestParsePrimaryNumber(t *testing.T) {
	toks, err := Tokenize("3.14")
	assert.NoError(t, err)
	p := NewParser(toks, &RunContext{})
	expr, err := p.parsePrimary()
	assert.NoError(t, err)
	lit, ok := expr.(*Literal)
	assert.True(t, ok)
	assert.EqualValues(t, 3.14, lit.Value)
}

func TestParsePrimaryString(t *testing.T) {
	toks, err := Tokenize(`"hello"`)
	assert.NoError(t, err)
	p := NewParser(toks, &RunContext{})
	expr, err := p.parsePrimary()
	assert.NoError(t, err)
	lit, ok := expr.(*Literal)
	assert.True(t, ok)
	assert.Equal(t, "hello", lit.Value)
}

func TestParsePrimaryBool(t *testing.T) {
	toks, err := Tokenize("false")
	assert.NoError(t, err)
	p := NewParser(toks, &RunContext{})
	expr, err := p.parsePrimary()
	assert.NoError(t, err)
	lit, ok := expr.(*Literal)
	assert.True(t, ok)
	assert.Equal(t, false, lit.Value.(bool))
}

func TestParsePrimaryIdentifier(t *testing.T) {
	toks, err := Tokenize("foo")
	assert.NoError(t, err)
	p := NewParser(toks, &RunContext{})
	expr, err := p.parsePrimary()
	assert.NoError(t, err)
	id, ok := expr.(*Identifier)
	assert.True(t, ok)
	assert.Equal(t, "foo", id.Name)
}

func TestParsePrimaryFunctionExpr(t *testing.T) {
	toks, err := Tokenize("function(a) { }")
	assert.NoError(t, err)
	p := NewParser(toks, &RunContext{})
	expr, err := p.parsePrimary()
	assert.NoError(t, err)
	fn, ok := expr.(*FunctionDecl)
	assert.True(t, ok)
	assert.Nil(t, fn.Name)
	assert.Len(t, fn.Params, 1)
}

func TestParsePrimaryNewExpr(t *testing.T) {
	toks, err := Tokenize("new Obj(1)")
	assert.NoError(t, err)
	p := NewParser(toks, &RunContext{})
	expr, err := p.parsePrimary()
	assert.NoError(t, err)
	ne, ok := expr.(*NewExpr)
	assert.True(t, ok)
	ident, ok := ne.Callee.(*Identifier)
	assert.True(t, ok)
	assert.Equal(t, "Obj", ident.Name)

	assert.Len(t, ne.Arguments, 1)
}

func TestParseObjectPattern(t *testing.T) {
	code := "{a, b, c}"
	toks, err := Tokenize(code)
	assert.NoError(t, err)
	p := NewParser(toks, &RunContext{})
	expr, err := p.parseObjectPattern()
	assert.NoError(t, err)
	obj, ok := expr.(*ObjectLiteral)
	assert.True(t, ok)
	// parseObjectPattern now collects Properties; expect 3 entries
	assert.Len(t, obj.Properties, 3)
}

// === parser_literals.go 解构模式 pattern 解析测试 ===
func TestParseObjectPatternCases(t *testing.T) {
	cases := []struct {
		code    string
		expectA float64
		expectB float64
		desc    string
		wantErr bool
	}{
		// 基本用法
		{"let obj = {a: 1, b: 2}; let {a, b} = obj; [a, b];", 1, 2, "basic object pattern", false},
		// 带逗号结尾
		{"let obj = {a: 3, b: 4}; let {a, b,} = obj; [a, b];", 3, 4, "object pattern with trailing comma", false},
		// 空对象
		{"let obj = {a: 1}; let {} = obj; 42;", 42, 0, "empty object pattern", false},
		// 非 identifier 错误
		{"let obj = {a: 1}; let {a, 1} = obj;", 0, 0, "object pattern with non-identifier", true},
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
			if c.desc == "empty object pattern" {
				if result.ToNumber() != 42 {
					t.Errorf("%s: expect 42, got %v", c.desc, result.ToString())
				}
				continue
			}
			arr := result.ToObject()
			if arr["0"].ToNumber() != c.expectA || arr["1"].ToNumber() != c.expectB {
				t.Errorf("%s: expect [%v, %v], got %v", c.desc, c.expectA, c.expectB, result.ToString())
			}
		}
	}
}

func TestParseArrayPattern(t *testing.T) {
	cases := []struct {
		code    string
		expectX float64
		expectY float64
		desc    string
		wantErr bool
	}{
		// 基本用法
		{"let arr = [1, 2]; let [x, y] = arr; [x, y];", 1, 2, "basic array pattern", false},
		// 带逗号结尾
		{"let arr = [3, 4]; let [x, y,] = arr; [x, y];", 3, 4, "array pattern with trailing comma", false},
		// 空数组
		{"let arr = [1]; let [] = arr; 99;", 99, 0, "empty array pattern", false},
		// 非 identifier 错误
		{"let arr = [1]; let [x, 1] = arr;", 0, 0, "array pattern with non-identifier", true},
	}
	for _, c := range cases {
		if c.wantErr {
			_, err := runJSWithError(t, c.code)
			if err == nil {
				t.Errorf("%s: expect error, got nil", c.desc)
			}
		} else {
			result := runJS(t, c.code)
			if c.desc == "empty array pattern" {
				if result.ToNumber() != 99 {
					t.Errorf("%s: expect 99, got %v", c.desc, result.ToString())
				}
				continue
			}
			arr := result.ToObject()
			if arr["0"].ToNumber() != c.expectX || arr["1"].ToNumber() != c.expectY {
				t.Errorf("%s: expect [%v, %v], got %v", c.desc, c.expectX, c.expectY, result.ToString())
			}
		}
	}
}
