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

func TestParseClassStatement(t *testing.T) {
	// Basic class without extends
	code := `class Animal {
	  constructor(name) {
	    this.name = name;
	  }
	  speak() {
	    return this.name + " makes a noise.";
	  }
	}`
	toks, err := Tokenize(code)
	assert.NoError(t, err)
	p := NewParser(toks, &RunContext{global: NewScope(nil), sourceLines: &[]string{code}})
	stmt, err := p.ParseStatement()
	assert.NoError(t, err)
	cls, ok := stmt.(*ClassDecl)
	assert.True(t, ok)
	assert.Equal(t, "Animal", cls.Name.Name)
	assert.Nil(t, cls.SuperClass)
	assert.Len(t, cls.Body, 2)
	// constructor method
	method := cls.Body[0]
	assert.Equal(t, "constructor", method.Key.Name)
	assert.Len(t, method.Value.Params, 1)
	// speak method
	method = cls.Body[1]
	assert.Equal(t, "speak", method.Key.Name)

	// Class with extends
	code2 := `class Dog extends Animal {
	  constructor(name) {
	    super(name);
	  }
	  speak() {
	    return this.name + " barks.";
	  }
	}`
	toks2, err := Tokenize(code2)
	assert.NoError(t, err)
	p2 := NewParser(toks2, &RunContext{global: NewScope(nil), sourceLines: &[]string{code2}})
	stmt2, err := p2.ParseStatement()
	assert.NoError(t, err)
	cls2, ok := stmt2.(*ClassDecl)
	assert.True(t, ok)
	assert.Equal(t, "Dog", cls2.Name.Name)
	assert.NotNil(t, cls2.SuperClass)
	ident, ok := cls2.SuperClass.(*Identifier)
	assert.True(t, ok)
	assert.Equal(t, "Animal", ident.Name)
	assert.Len(t, cls2.Body, 2)
	// constructor
	method = cls2.Body[0]
	assert.Equal(t, "constructor", method.Key.Name)
	assert.Len(t, method.Value.Params, 1)
	// speak
	method = cls2.Body[1]
	assert.Equal(t, "speak", method.Key.Name)
}

func TestParseClassInheritance(t *testing.T) {
	code := `class Animal {
	  constructor(name) {
	    this.name = name;
	  }
	  speak() {
	    return this.name + " makes a noise.";
	  }
	}
	class Dog extends Animal {
	  constructor(name) {
	    super(name);
	  }
	  speak() {
	    return this.name + " barks.";
	  }
	}`
	toks, err := Tokenize(code)
	assert.NoError(t, err)
	p := NewParser(toks, &RunContext{global: NewScope(nil), sourceLines: &[]string{code}})

	// 解析第一个 class
	stmt, err := p.ParseStatement()
	assert.NoError(t, err)
	cls, ok := stmt.(*ClassDecl)
	assert.True(t, ok)
	assert.Equal(t, "Animal", cls.Name.Name)
	assert.Nil(t, cls.SuperClass)
	assert.Len(t, cls.Body, 2)
	assert.Equal(t, "constructor", cls.Body[0].Key.Name)
	assert.Equal(t, "speak", cls.Body[1].Key.Name)

	// 解析第二个 class
	stmt2, err := p.ParseStatement()
	assert.NoError(t, err)
	cls2, ok := stmt2.(*ClassDecl)
	assert.True(t, ok)
	assert.Equal(t, "Dog", cls2.Name.Name)
	assert.NotNil(t, cls2.SuperClass)
	assert.IsType(t, &Identifier{}, cls2.SuperClass)
	assert.Equal(t, "Animal", cls2.SuperClass.(*Identifier).Name)
	assert.Len(t, cls2.Body, 2)
	assert.Equal(t, "constructor", cls2.Body[0].Key.Name)
	assert.Equal(t, "speak", cls2.Body[1].Key.Name)
}
