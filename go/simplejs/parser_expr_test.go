package simplejs

import (
	"strconv"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestParseAssignmentExpr(t *testing.T) {
	toks, err := Tokenize("x = 5")
	assert.NoError(t, err)
	ctx := &RunContext{global: NewScope(nil)}
	p := NewParser(toks, ctx)
	expr, err := p.ParseExpression()
	assert.NoError(t, err)
	assign, ok := expr.(*AssignmentExpr)
	assert.True(t, ok)
	left, ok := assign.Left.(*Identifier)
	assert.True(t, ok)
	assert.Equal(t, "x", left.Name)
	lit, ok := assign.Right.(*Literal)
	assert.True(t, ok)
	assert.EqualValues(t, 5, lit.Value)
}

func TestParseLogicalOrAnd(t *testing.T) {
	toks, err := Tokenize("a || b && c")
	assert.NoError(t, err)
	ctx := &RunContext{global: NewScope(nil)}
	p := NewParser(toks, ctx)
	expr, err := p.ParseExpression()
	assert.NoError(t, err)
	orExpr, ok := expr.(*BinaryExpr)
	assert.True(t, ok)
	assert.Equal(t, "||", orExpr.Op)
	rightAnd, ok := orExpr.Right.(*BinaryExpr)
	assert.True(t, ok)
	assert.Equal(t, "&&", rightAnd.Op)
}

func TestParseEqualityAndRelational(t *testing.T) {
	toks, err := Tokenize("a == b != c < d >= e")
	assert.NoError(t, err)
	ctx := &RunContext{global: NewScope(nil)}
	p := NewParser(toks, ctx)
	expr, err := p.ParseExpression()
	assert.NoError(t, err)
	top, ok := expr.(*BinaryExpr)
	assert.True(t, ok)
	// equality chain
	leftEq, ok := top.Left.(*BinaryExpr)
	assert.True(t, ok)
	assert.Equal(t, "==", leftEq.Op)
	assert.Equal(t, "!=", top.Op)
	// relational chain inside right side
	relTop, ok := top.Right.(*BinaryExpr)
	assert.True(t, ok)
	assert.Equal(t, ">=", relTop.Op)
	innerRel, ok := relTop.Left.(*BinaryExpr)
	assert.True(t, ok)
	assert.Equal(t, "<", innerRel.Op)
}

func TestParseUnaryAndPostfix(t *testing.T) {
	toks, err := Tokenize("!x--")
	assert.NoError(t, err)
	ctx := &RunContext{global: NewScope(nil)}
	p := NewParser(toks, ctx)
	expr, err := p.ParseExpression()
	assert.NoError(t, err)
	// 先识别后缀更新表达式
	upd, ok := expr.(*UpdateExpr)
	assert.True(t, ok)
	assert.Equal(t, "--", upd.Op)
	assert.False(t, upd.Prefix)
	// 检查更新对象是 一元表达式
	un, ok := upd.Argument.(*UnaryExpr)
	assert.True(t, ok)
	assert.Equal(t, "!", un.Op)
	// 最终操作对象
	ident, ok := un.X.(*Identifier)
	assert.True(t, ok)
	assert.Equal(t, "x", ident.Name)
}

func TestParseAddSubMulDiv(t *testing.T) {
	toks, err := Tokenize("1 + 2 * 3 - 4 / 2")
	assert.NoError(t, err)
	ctx := &RunContext{global: NewScope(nil)}
	p := NewParser(toks, ctx)
	expr, err := p.ParseExpression()
	assert.NoError(t, err)
	root, ok := expr.(*BinaryExpr)
	assert.True(t, ok)
	assert.Equal(t, "-", root.Op)
	leftAdd, ok := root.Left.(*BinaryExpr)
	assert.True(t, ok)
	assert.Equal(t, "+", leftAdd.Op)
	mul, ok := leftAdd.Right.(*BinaryExpr)
	assert.True(t, ok)
	assert.Equal(t, "*", mul.Op)
	div, ok := root.Right.(*BinaryExpr)
	assert.True(t, ok)
	assert.Equal(t, "/", div.Op)
}

func TestParseTernary(t *testing.T) {
	toks, err := Tokenize("cond ? a : b")
	assert.NoError(t, err)
	ctx := &RunContext{global: NewScope(nil)}
	p := NewParser(toks, ctx)
	expr, err := p.ParseExpression()
	assert.NoError(t, err)
	condExpr, ok := expr.(*ConditionalExpr)
	assert.True(t, ok)
	ident, ok := condExpr.Test.(*Identifier)
	assert.True(t, ok)
	assert.Equal(t, "cond", ident.Name)
	cons, ok := condExpr.Consequent.(*Identifier)
	assert.True(t, ok)
	assert.Equal(t, "a", cons.Name)
	alt, ok := condExpr.Alternate.(*Identifier)
	assert.True(t, ok)
	assert.Equal(t, "b", alt.Name)
}

// === parser_expressions.go 赋值表达式解析测试 ===
func TestAssignmentExpressions(t *testing.T) {
	cases := []struct {
		code    string
		expect  interface{} // 允许 float64 或 string
		desc    string
		wantErr bool
	}{
		// 普通变量赋值
		{"let x = 1; x = 2; x;", 2.0, "simple variable assignment", false},
		// 右结合赋值
		{"let a, b, c; a = b = c = 5; [a, b, c];", []float64{5, 5, 5}, "right associative assignment", false},
		// 对象属性赋值
		{"let obj = {x: 1}; obj.x = 42; obj.x;", 42.0, "object property assignment (dot)", false},
		{"let obj = {}; obj['y'] = 99; obj.y;", 99.0, "object property assignment (bracket)", false},
		// 数组元素赋值及 length
		{"let arr = [1,2]; arr[1] = 7; arr[1];", 7.0, "array element assignment", false},
		{"let arr = [1,2]; arr[2] = 9; arr.length;", 3.0, "array length auto-update", false},
		// 错误：赋值给字面量
		{"5 = 1;", nil, "assign to literal should error", true},
		// 错误：赋值给不存在对象属性
		{"foo.bar = 1;", nil, "assign to property of undefined object should error", true},
		// 错误：赋值给非对象
		{"let x = 1; x.y = 2;", nil, "assign to property of non-object should error", true},
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
			if arr, ok := c.expect.([]float64); ok {
				robj := result.ToObject()
				for i, v := range arr {
					if robj[strconv.Itoa(i)].ToNumber() != v {
						t.Errorf("%s: expect arr[%d]=%v, got %v", c.desc, i, v, robj[strconv.Itoa(i)].ToNumber())
					}
				}
			} else if num, ok := c.expect.(float64); ok {
				if result.ToNumber() != num {
					t.Errorf("%s: expect %v, got %v", c.desc, num, result.ToString())
				}
			}
		}
	}
}
