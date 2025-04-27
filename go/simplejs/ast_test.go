package simplejs

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestIdentifierAndLiteral(t *testing.T) {
	id := &Identifier{Name: "foo", Line: 1, Col: 2}
	lit := &Literal{Value: 123, Line: 2, Col: 3}

	assert.Equal(t, "foo", id.Name)
	assert.Equal(t, 123, lit.Value)
	line, _ := id.Pos()
	assert.Equal(t, 1, line)
	_, col := lit.Pos()
	assert.Equal(t, 3, col)
}

func TestBinaryAndAssignmentExpr(t *testing.T) {
	left := &Identifier{Name: "a", Line: 1, Col: 1}
	right := &Literal{Value: 42, Line: 1, Col: 3}
	bin := &BinaryExpr{Op: "+", Left: left, Right: right, Line: 1, Col: 2}
	assign := &AssignmentExpr{Operator: "=", Left: left, Right: right, Line: 2, Col: 2}

	assert.Equal(t, "+", bin.Op)
	assert.Equal(t, "a", bin.Left.(*Identifier).Name)
	assert.Equal(t, 42, bin.Right.(*Literal).Value)
	assert.Equal(t, "=", assign.Operator)
}

func TestConditionalExpr(t *testing.T) {
	cond := &ConditionalExpr{
		Test:       &Literal{Value: true, Line: 1, Col: 1},
		Consequent: &Literal{Value: 1, Line: 1, Col: 2},
		Alternate:  &Literal{Value: 0, Line: 1, Col: 3},
		Line:       1, Col: 1,
	}
	assert.Equal(t, true, cond.Test.(*Literal).Value)
	assert.Equal(t, 1, cond.Consequent.(*Literal).Value)
	assert.Equal(t, 0, cond.Alternate.(*Literal).Value)
}

func TestArrayAndObjectLiteral(t *testing.T) {
	arr := &ArrayLiteral{
		Elements: []Expression{
			&Literal{Value: 1}, &Literal{Value: 2},
		},
		Line: 1, Col: 1,
	}
	obj := &ObjectLiteral{
		Properties: []*Property{
			{Key: &Identifier{Name: "x"}, Value: &Literal{Value: 42}, Kind: "init"},
		},
		Line: 2, Col: 2,
	}
	assert.Len(t, arr.Elements, 2)
	assert.Equal(t, "x", obj.Properties[0].Key.(*Identifier).Name)
	assert.Equal(t, 42, obj.Properties[0].Value.(*Literal).Value)
}

func TestTemplateLiteral(t *testing.T) {
	tpl := &TemplateLiteral{
		Quasis:      []string{"Hello, ", "!"},
		Expressions: []Expression{&Identifier{Name: "name"}},
		Line:        1, Col: 1,
	}
	assert.Equal(t, "Hello, ", tpl.Quasis[0])
	assert.Equal(t, "name", tpl.Expressions[0].(*Identifier).Name)
}

func TestFunctionDeclAndCall(t *testing.T) {
	fn := &FunctionDecl{
		Name:   &Identifier{Name: "add"},
		Params: []*Identifier{{Name: "a"}, {Name: "b"}},
		Body:   &BlockStmt{Body: []Statement{}},
		Line:   1, Col: 1,
	}
	call := &CallExpr{
		Callee:    fn,
		Arguments: []Expression{&Literal{Value: 1}, &Literal{Value: 2}},
		Line:      2, Col: 2,
	}
	assert.Equal(t, "add", fn.Name.Name)
	assert.Len(t, fn.Params, 2)
	assert.Len(t, call.Arguments, 2)
}

func TestSwitchAndClass(t *testing.T) {
	sw := &SwitchStmt{
		Discriminant: &Identifier{Name: "x"},
		Cases: []*CaseClause{
			{Test: &Literal{Value: 1}, Consequent: []Statement{}},
			{Test: nil, Consequent: []Statement{}}, // default
		},
		Line: 1, Col: 1,
	}
	class := &ClassDecl{
		Name:       &Identifier{Name: "A"},
		SuperClass: &Identifier{Name: "B"},
		Body: []*MethodDef{
			{Key: &Identifier{Name: "foo"}, Value: &FunctionDecl{}},
		},
		Line: 2, Col: 2,
	}
	assert.Equal(t, "x", sw.Discriminant.(*Identifier).Name)
	assert.Nil(t, sw.Cases[1].Test)
	assert.Equal(t, "A", class.Name.Name)
	assert.Equal(t, "B", class.SuperClass.(*Identifier).Name)
	assert.Equal(t, "foo", class.Body[0].Key.Name)
}

func TestVarLetConst(t *testing.T) {
	js := `
		var a = 1;
		let b = 2;
		const c = a + b;
	`
	tokens, err := Tokenize(js)
	assert.NoError(t, err)
	ctx := &RunContext{global: NewScope(nil)}
	parser := NewParser(tokens, ctx)
	program, err := parser.ParseProgram()
	assert.NoError(t, err)
	assert.NotNil(t, program)

	assert.Equal(t, 3, len(program.Body))

	decl1, ok := program.Body[0].(*VarDecl)
	assert.True(t, ok)
	assert.Equal(t, "var", decl1.Kind)
	if ident, ok := decl1.Name.(*Identifier); ok {
		assert.Equal(t, "a", ident.Name)
	} else {
		t.Errorf("decl1.Name is not Identifier, got %T", decl1.Name)
	}
	lit1, ok := decl1.Init.(*Literal)
	assert.True(t, ok)
	assert.EqualValues(t, 1, lit1.Value)

	decl2, ok := program.Body[1].(*VarDecl)
	assert.True(t, ok)
	assert.Equal(t, "let", decl2.Kind)
	if ident, ok := decl2.Name.(*Identifier); ok {
		assert.Equal(t, "b", ident.Name)
	} else {
		t.Errorf("decl2.Name is not Identifier, got %T", decl2.Name)
	}
	lit2, ok := decl2.Init.(*Literal)
	assert.True(t, ok)
	assert.EqualValues(t, 2, lit2.Value)

	decl3, ok := program.Body[2].(*VarDecl)
	assert.True(t, ok)
	assert.Equal(t, "const", decl3.Kind)
	if ident, ok := decl3.Name.(*Identifier); ok {
		assert.Equal(t, "c", ident.Name)
	} else {
		t.Errorf("decl3.Name is not Identifier, got %T", decl3.Name)
	}
	bin, ok := decl3.Init.(*BinaryExpr)
	assert.True(t, ok)
	assert.Equal(t, "+", bin.Op)
	left, ok := bin.Left.(*Identifier)
	assert.True(t, ok)
	assert.Equal(t, "a", left.Name)
	right, ok := bin.Right.(*Identifier)
	assert.True(t, ok)
	assert.Equal(t, "b", right.Name)
}

func TestIfElseWhileFor(t *testing.T) {
	js := `
		if (a > 1) { b = 2; } else { b = 3; }
		while (b < 10) { b++; }
		for (let i = 0; i < 5; i++) { b += i; }
	`
	tokens, err := Tokenize(js)
	assert.NoError(t, err)
	ctx := &RunContext{global: NewScope(nil)}
	parser := NewParser(tokens, ctx)
	program, err := parser.ParseProgram()
	assert.NoError(t, err)
	assert.NotNil(t, program)

	assert.Equal(t, 3, len(program.Body))

	ifStmt, ok := program.Body[0].(*IfStmt)
	assert.True(t, ok)
	assert.NotNil(t, ifStmt.Test)
	assert.NotNil(t, ifStmt.Consequent)
	assert.NotNil(t, ifStmt.Alternate)

	whileStmt, ok := program.Body[1].(*WhileStmt)
	assert.True(t, ok)
	assert.NotNil(t, whileStmt.Test)
	assert.NotNil(t, whileStmt.Body)

	forStmt, ok := program.Body[2].(*ForStmt)
	assert.True(t, ok)
	assert.NotNil(t, forStmt.Init)
	assert.NotNil(t, forStmt.Test)
	assert.NotNil(t, forStmt.Update)
	assert.NotNil(t, forStmt.Body)
}

// Test AST ReturnStmt node
func TestReturnStmt(t *testing.T) {
	ret := &ReturnStmt{Argument: &Literal{Value: 42, Line: 5, Col: 10}, Line: 1, Col: 2}
	assert.Equal(t, 42, ret.Argument.(*Literal).Value)
	line, col := ret.Pos()
	assert.Equal(t, 1, line)
	assert.Equal(t, 2, col)
}

// Test AST ThrowStmt node
func TestThrowStmtAST(t *testing.T) {
	thr := &ThrowStmt{Argument: &Literal{Value: "err", Line: 3, Col: 4}, Line: 3, Col: 4}
	assert.Equal(t, "err", thr.Argument.(*Literal).Value)
	line, col := thr.Pos()
	assert.Equal(t, 3, line)
	assert.Equal(t, 4, col)
}

// Test AST TryCatchStmt node
func TestTryCatchStmtAST(t *testing.T) {
	tryBlk := &BlockStmt{Body: []Statement{}, Line: 1, Col: 1}
	catchParam := &Identifier{Name: "e", Line: 1, Col: 5}
	catchBlk := &BlockStmt{Body: []Statement{}, Line: 2, Col: 1}
	finBlk := &BlockStmt{Body: []Statement{}, Line: 3, Col: 1}
	tc := &TryCatchStmt{TryBlock: tryBlk, CatchParam: catchParam, CatchBlock: catchBlk, FinallyBlock: finBlk, Line: 1, Col: 1}
	assert.Equal(t, tryBlk, tc.TryBlock)
	assert.Equal(t, catchParam, tc.CatchParam)
	assert.Equal(t, catchBlk, tc.CatchBlock)
	assert.Equal(t, finBlk, tc.FinallyBlock)
	line, col := tc.Pos()
	assert.Equal(t, 1, line)
	assert.Equal(t, 1, col)
}

// Test AST BreakStmt and ContinueStmt nodes
func TestBreakAndContinueStmtAST(t *testing.T) {
	br := &BreakStmt{Line: 5, Col: 6}
	assert.Equal(t, 5, br.Line)
	line, col := br.Pos()
	assert.Equal(t, 5, line)
	assert.Equal(t, 6, col)

	cont := &ContinueStmt{Label: &Identifier{Name: "label"}, Line: 7, Col: 8}
	assert.Equal(t, "label", cont.Label.Name)
	line2, col2 := cont.Pos()
	assert.Equal(t, 7, line2)
	assert.Equal(t, 8, col2)
}

// Test Eval arithmetic expressions
func TestEvalArithmetic(t *testing.T) {
	ctx := NewContext(0)
	result, err := ctx.Eval("1 + 2 * 3;")
	assert.NoError(t, err)
	assert.EqualValues(t, 7, result.ToNumber())
}

// Test Eval variable binding
func TestEvalVarBinding(t *testing.T) {
	ctx := NewContext(0)
	result, err := ctx.Eval("var x = 10; x;")
	assert.NoError(t, err)
	assert.EqualValues(t, 10, result.ToNumber())
}

// Test AST DeleteExpr node
func TestDeleteExprAST(t *testing.T) {
	del := &DeleteExpr{Argument: &Identifier{Name: "x"}, Line: 1, Col: 2}
	assert.Equal(t, "x", del.Argument.(*Identifier).Name)
	line, col := del.Pos()
	assert.Equal(t, 1, line)
	assert.Equal(t, 2, col)
}

// Test AST UpdateExpr node
func TestUpdateExprAST(t *testing.T) {
	upd := &UpdateExpr{Op: "++", Argument: &Identifier{Name: "i"}, Prefix: true, Line: 2, Col: 3}
	assert.Equal(t, "++", upd.Op)
	assert.True(t, upd.Prefix)
	arg := upd.Argument.(*Identifier)
	assert.Equal(t, "i", arg.Name)
	line, col := upd.Pos()
	assert.Equal(t, 2, line)
	assert.Equal(t, 3, col)
}

// Test AST SpreadElement node
func TestSpreadElementAST(t *testing.T) {
	sp := &SpreadElement{Argument: &Identifier{Name: "rest"}, Line: 4, Col: 5}
	assert.Equal(t, "rest", sp.Argument.(*Identifier).Name)
	line, col := sp.Pos()
	assert.Equal(t, 4, line)
	assert.Equal(t, 5, col)
}

// Test AST ArrowFunctionExpr node
func TestArrowFunctionExprAST(t *testing.T) {
	arrow := &ArrowFunctionExpr{
		Params: []*Identifier{{Name: "a", Line: 1, Col: 1}, {Name: "b", Line: 1, Col: 2}},
		Body:   &Literal{Value: 100, Line: 1, Col: 3},
		Async:  true,
		Line:   6, Col: 7,
	}
	assert.Len(t, arrow.Params, 2)
	expr, ok := arrow.Body.(*Literal)
	assert.True(t, ok)
	assert.Equal(t, 100, expr.Value)
	assert.True(t, arrow.Async)
	line, col := arrow.Pos()
	assert.Equal(t, 6, line)
	assert.Equal(t, 7, col)
}
