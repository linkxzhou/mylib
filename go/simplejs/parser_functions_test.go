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
	assert.True(t, ok)
	fn, ok := exprStmt.Expr.(*FunctionDecl)
	assert.True(t, ok)
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
	assert.True(t, ok)
	fn, ok := exprStmt.Expr.(*FunctionDecl)
	assert.True(t, ok)
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
	assert.True(t, ok)
	ident, ok := ne.Callee.(*Identifier)
	assert.True(t, ok)
	assert.Equal(t, "Obj", ident.Name)
	assert.Len(t, ne.Arguments, 2)
}

func TestParseFunctionExprWithParams(t *testing.T) {
	code := "function(a, b) { return a + b; }"
	toks, err := Tokenize(code)
	assert.NoError(t, err)
	ctx := &RunContext{global: NewScope(nil)}
	p := NewParser(toks, ctx)
	stmt, err := p.ParseStatement()
	assert.NoError(t, err)
	exprStmt, ok := stmt.(*ExpressionStmt)
	assert.True(t, ok)
	fn, ok := exprStmt.Expr.(*FunctionDecl)
	assert.True(t, ok)
	assert.Nil(t, fn.Name)
	assert.Len(t, fn.Params, 2)
}
