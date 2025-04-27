package simplejs

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestParseException_Error(t *testing.T) {
	ex := ParseException{Msg: "unexpected", Line: 5, Column: 10, Code: "let x = ;"}
	s := ex.Error()
	assert.Contains(t, s, "line 5")
	assert.Contains(t, s, "column 10")
	assert.Contains(t, s, "unexpected")
	assert.Contains(t, s, "let x = ;")
}

func TestNewParserInit_Core(t *testing.T) {
	toks := []Token{{Type: TokNumber, Literal: "1"}}
	ctx := &RunContext{global: NewScope(nil)}
	p := NewParser(toks, ctx)
	assert.Equal(t, toks, p.tokens)
	assert.Equal(t, 0, p.pos)
	assert.Equal(t, ctx, p.ctx)
}

func TestPeekNextExpect(t *testing.T) {
	toks := []Token{
		{Type: TokNumber, Literal: "1", Line: 1},
		{Type: TokPlus, Literal: "+", Line: 1},
		{Type: TokNumber, Literal: "2", Line: 1},
		{Type: TokEOF, Literal: "", Line: 1},
	}
	p := NewParser(toks, &RunContext{})
	tok := p.peek()
	assert.Equal(t, "1", tok.Literal)
	tokN := p.next()
	assert.Equal(t, "1", tokN.Literal)
	tok2 := p.peek()
	assert.Equal(t, "+", tok2.Literal)
	tokExpect, err := p.expect(TokPlus)
	assert.NoError(t, err)
	assert.Equal(t, "+", tokExpect.Literal)
	_, err2 := p.expect(TokNumber)
	assert.NoError(t, err2)
	_, err3 := p.expect(TokSemicolon)
	assert.Error(t, err3)
	assert.Contains(t, err3.Error(), "expected")
	assert.Contains(t, err3.Error(), "got")
}

func TestCurrentLineInfo_Custom(t *testing.T) {
	lines := []string{"first line", "second foo line"}
	ctx := &RunContext{sourceLines: &lines}
	toks := []Token{{Type: TokIdentifier, Literal: "foo", Line: 2}}
	p := NewParser(toks, ctx)
	line, col, code := p.currentLineInfo()
	assert.Equal(t, 2, line)
	assert.Equal(t, strings.Index(lines[1], "foo")+1, col)
	assert.Equal(t, "second foo line", code)
}

// Test parsing a simple expression program
func TestParseProgram_SimpleExpression(t *testing.T) {
	code := "42;"
	lines := strings.Split(code, "\n")
	ctx := &RunContext{sourceLines: &lines}
	toks, err := Tokenize(code)
	assert.NoError(t, err)
	p := NewParser(toks, ctx)
	prog, err := p.ParseProgram()
	assert.NoError(t, err)
	assert.Len(t, prog.Body, 1)
	stmt, ok := prog.Body[0].(*ExpressionStmt)
	assert.True(t, ok)
	lit, ok := stmt.Expr.(*Literal)
	assert.True(t, ok)
	assert.EqualValues(t, 42, lit.Value)
}

// Test expect EOF token
func TestExpectEOF(t *testing.T) {
	toks := []Token{{Type: TokEOF, Literal: "", Line: 1}}
	p := NewParser(toks, &RunContext{})
	tok, err := p.expect(TokEOF)
	assert.NoError(t, err)
	assert.Equal(t, TokEOF, tok.Type)
}
