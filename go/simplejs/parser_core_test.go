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

// === parser_core.go 基础功能与异常测试 ===
func TestParseExceptionFormat(t *testing.T) {
	err := ParseException{Msg: "unexpected token", Line: 3, Code: "let x = ;"}
	msg := err.Error()
	if !strings.Contains(msg, "Parse error at line 3") || !strings.Contains(msg, "unexpected token") || !strings.Contains(msg, "let x = ;") {
		t.Errorf("ParseException format error: %s", msg)
	}
}

func TestNewParserInit_SimpleJS(t *testing.T) {
	toks := []Token{{Type: TokNumber, Literal: "1"}}
	ctx := &RunContext{global: NewScope(nil)}
	p := NewParser(toks, ctx)
	if len(p.tokens) != 1 || p.pos != 0 || p.ctx != ctx {
		t.Errorf("NewParser did not initialize fields correctly")
	}
}

func TestParseExpressionAndProgram(t *testing.T) {
	ctx := &RunContext{global: NewScope(nil)}
	// ParseExpression: 1+2
	toks, _ := Tokenize("1+2")
	p := NewParser(toks, ctx)
	expr, err := p.ParseExpression()
	if err != nil {
		t.Errorf("ParseExpression failed: err=%v", err)
	}
	progExpr := &Program{Body: []Statement{&ExpressionStmt{Expr: expr}}}
	v, err := progExpr.Eval(ctx)
	if err != nil || v.ToNumber() != 3 {
		t.Errorf("ParseExpression failed: got %v, err=%v", v.ToString(), err)
	}

	// ParseProgram: 多条语句和多余分号
	code := "let a = 1;; a = 2; a;"
	toks2, _ := Tokenize(code)
	p2 := NewParser(toks2, ctx)
	prog, err2 := p2.ParseProgram()
	assert.NoError(t, err2)
	val2, err2 := prog.Eval(ctx)
	assert.NoError(t, err2)
	if val2.ToNumber() != 2 {
		t.Errorf("ParseProgram failed: got %v, err=%v", val2.ToString(), err2)
	}

	// ParseProgram: 语法错误
	code3 := "let x = ;"
	toks3, _ := Tokenize(code3)
	p3 := NewParser(toks3, ctx)
	_, err3 := p3.ParseProgram()
	if err3 == nil {
		t.Error("ParseProgram should error on syntax error")
	} else if !strings.Contains(err3.Error(), "Parse error") {
		t.Errorf("ParseProgram error format: %v", err3)
	}
}

func TestParserPeekNextExpect(t *testing.T) {
	toks, _ := Tokenize("1 + 2")
	ctx := &RunContext{global: NewScope(nil)}
	p := NewParser(toks, ctx)
	if p.peek().Literal != "1" {
		t.Errorf("peek() should return first token")
	}
	tok := p.next()
	if tok.Literal != "1" || p.peek().Literal != "+" {
		t.Errorf("next() should advance to next token")
	}
	tok2, err := p.expect(TokPlus)
	if err != nil || tok2.Literal != "+" {
		t.Errorf("expect() should consume + token, got %v err=%v", tok2.Literal, err)
	}
	_, err2 := p.expect(TokNumber)
	if err2 != nil {
		t.Errorf("expect() should succeed for number token")
	}
	_, err3 := p.expect(TokSemicolon)
	if err3 == nil {
		t.Error("expect() should fail for missing token")
	}
}
