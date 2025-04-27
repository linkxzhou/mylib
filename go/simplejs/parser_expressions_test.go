package simplejs

import (
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
