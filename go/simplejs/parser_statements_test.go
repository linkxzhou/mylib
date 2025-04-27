package simplejs

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestParseReturnStmt(t *testing.T) {
	code := "return 42;"
	toks, err := Tokenize(code)
	assert.NoError(t, err)
	p := NewParser(toks, &RunContext{global: NewScope(nil), sourceLines: &[]string{code}})
	stmt, err := p.ParseStatement()
	assert.NoError(t, err)
	rtn, ok := stmt.(*ReturnStmt)
	assert.True(t, ok)
	lit, ok := rtn.Argument.(*Literal)
	assert.True(t, ok)
	assert.EqualValues(t, 42.0, lit.Value)
}

func TestParseThrowStmt(t *testing.T) {
	code := "throw \"err\";"
	toks, err := Tokenize(code)
	assert.NoError(t, err)
	p := NewParser(toks, &RunContext{global: NewScope(nil), sourceLines: &[]string{code}})
	stmt, err := p.ParseStatement()
	assert.NoError(t, err)
	thr, ok := stmt.(*ThrowStmt)
	assert.True(t, ok)
	lit, ok := thr.Argument.(*Literal)
	assert.True(t, ok)
	assert.Equal(t, "err", lit.Value)
}

func TestParseExpressionStmtDefault(t *testing.T) {
	code := "x + y;"
	toks, err := Tokenize(code)
	assert.NoError(t, err)
	p := NewParser(toks, &RunContext{global: NewScope(nil)})
	stmt, err := p.ParseStatement()
	assert.NoError(t, err)
	_, ok := stmt.(*ExpressionStmt)
	assert.True(t, ok)
}

func TestParseNewStmt(t *testing.T) {
	code := "new Obj();"
	toks, err := Tokenize(code)
	assert.NoError(t, err)
	p := NewParser(toks, &RunContext{global: NewScope(nil)})
	stmt, err := p.ParseStatement()
	assert.NoError(t, err)
	exprStmt, ok := stmt.(*ExpressionStmt)
	assert.True(t, ok)
	newExpr, ok := exprStmt.Expr.(*NewExpr)
	assert.True(t, ok)
	ident, ok := newExpr.Callee.(*Identifier)
	assert.True(t, ok)
	assert.Equal(t, "Obj", ident.Name)
}
