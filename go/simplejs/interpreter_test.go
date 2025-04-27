package simplejs

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

// helper to parse and eval code
func evalCode(code string) JSValue {
	toks, err := Tokenize(code)
	if err != nil {
		panic(err)
	}
	ctx := &RunContext{global: NewScope(nil), sourceLines: &[]string{code}}
	p := NewParser(toks, ctx)
	prog, err := p.ParseProgram()
	if err != nil {
		panic(err)
	}
	val, err := prog.Eval(ctx)
	if err != nil {
		panic(err)
	}
	return val
}

func TestEvalLiteralAndBinary(t *testing.T) {
	// literal
	v1 := evalCode("42;")
	assert.Equal(t, 42.0, v1.ToNumber())
	// binary
	v2 := evalCode("1+2*3;")
	assert.Equal(t, 7.0, v2.ToNumber())
}

func TestEvalIfElse(t *testing.T) {
	code := "if (true) { 1; } else { 2; }"
	v := evalCode(code)
	assert.Equal(t, 1.0, v.ToNumber())
}

func TestEvalTryCatch(t *testing.T) {
	code := "try { throw 100; } catch (e) { e; }"
	v := evalCode(code)
	assert.Equal(t, 100.0, v.ToNumber())
}

func TestEvalVarDeclaration(t *testing.T) {
	v := evalCode("var x = 5; x;")
	assert.Equal(t, 5.0, v.ToNumber())
}

func TestEvalArithmeticAndComparison(t *testing.T) {
	assert.Equal(t, 3.0, evalCode("5-2;").ToNumber())
	assert.Equal(t, 2.0, evalCode("6/3;").ToNumber())
	assert.True(t, evalCode("5<10;").ToBool())
	assert.False(t, evalCode("5>10;").ToBool())
}

func TestEvalLogicalAndUnary(t *testing.T) {
	assert.True(t, evalCode("true;").ToBool())
	assert.False(t, evalCode("false;").ToBool())
	assert.False(t, evalCode("!true;").ToBool())
	assert.True(t, evalCode("!false;").ToBool())
}

func TestEvalEquality(t *testing.T) {
	assert.True(t, evalCode("5==5;").ToBool())
	assert.False(t, evalCode("5!=5;").ToBool())
	assert.True(t, evalCode("5===5;").ToBool())
	// strict inequality: types differ -> true
	assert.True(t, evalCode("5!==\"5\";").ToBool())
}

func TestEvalStringAndConcat(t *testing.T) {
	// string literal
	assert.Equal(t, "hello", evalCode("\"hello\";").ToString())
	// concatenation
	assert.Equal(t, "foo bar", evalCode("\"foo\"+\" bar\";").ToString())
}

func TestEvalUnaryNegation(t *testing.T) {
	// numeric negation
	assert.Equal(t, -5.0, evalCode("-5;").ToNumber())
	// logical not on number
	// 0 is falsey, !0 should be true
	assert.True(t, evalCode("!0;").ToBool())
}
