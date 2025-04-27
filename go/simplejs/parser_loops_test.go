package simplejs

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestParseWhile(t *testing.T) {
	code := "while (x < 1) { x = x + 1; }"
	toks, err := Tokenize(code)
	assert.NoError(t, err)
	ctx := &RunContext{global: NewScope(nil)}
	p := NewParser(toks, ctx)
	stmt, err := p.ParseStatement()
	assert.NoError(t, err)
	ws, ok := stmt.(*WhileStmt)
	assert.True(t, ok)
	testExpr, ok := ws.Test.(*BinaryExpr)
	assert.True(t, ok)
	assert.Equal(t, "<", testExpr.Op)
	body, ok := ws.Body.(*BlockStmt)
	assert.True(t, ok)
	// body contains one ExpressionStmt
	_, ok = body.Body[0].(*ExpressionStmt)
	assert.True(t, ok)
}

func TestParseFor(t *testing.T) {
	code := "for (var i = 0; i < 2; i++) { doSomething(); }"
	toks, err := Tokenize(code)
	assert.NoError(t, err)
	ctx := &RunContext{global: NewScope(nil)}
	p := NewParser(toks, ctx)
	stmt, err := p.ParseStatement()
	assert.NoError(t, err)
	fs, ok := stmt.(*ForStmt)
	assert.True(t, ok)
	// init
	vd, ok := fs.Init.(*VarDecl)
	assert.True(t, ok)
	assert.Equal(t, "var", vd.Kind)
	// test
	testExpr, ok := fs.Test.(*BinaryExpr)
	assert.True(t, ok)
	assert.Equal(t, "<", testExpr.Op)
	// update
	upd, ok := fs.Update.(*UpdateExpr)
	assert.True(t, ok)
	assert.Equal(t, "++", upd.Op)
	// body
	body, ok := fs.Body.(*BlockStmt)
	assert.True(t, ok)
	// body contains ExpressionStmt for call
	exprStmt, ok := body.Body[0].(*ExpressionStmt)
	assert.True(t, ok)
	// 调用表达式作为 CallExpr 节点
	call, ok := exprStmt.Expr.(*CallExpr)
	if !assert.True(t, ok) {
		return
	}
	ident, ok := call.Callee.(*Identifier)
	if !assert.True(t, ok) {
		return
	}
	assert.Equal(t, "doSomething", ident.Name)
}
