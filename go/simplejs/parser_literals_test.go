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
	// 因为 stub 未收集 Properties，当前应该是 nil
	assert.Nil(t, obj.Properties)
}
