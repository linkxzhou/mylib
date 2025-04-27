package simplejs

// Node 是所有 AST 节点的基础接口
type Node interface {
	Pos() (line, col int)
}

// ========== 程序结构 ==========

// Program 表示整个 JS 程序（文件）
type Program struct {
	Body []Statement
}

func (p *Program) Pos() (int, int) {
	if len(p.Body) > 0 {
		return p.Body[0].Pos()
	}
	return 0, 0
}

// ========== 语句和表达式 ==========

type Statement interface {
	Node
	stmtNode()
}

type Expression interface {
	Node
	exprNode()
}

// ========== 标识符和字面量 ==========

type Identifier struct {
	Name string
	Line int
	Col  int
}

func (i *Identifier) Pos() (int, int) { return i.Line, i.Col }
func (i *Identifier) exprNode()       {}

type Literal struct {
	Value interface{}
	Line  int
	Col   int
}

func (l *Literal) Pos() (int, int) { return l.Line, l.Col }
func (l *Literal) exprNode()       {}

// ========== 表达式节点 ==========

type BinaryExpr struct {
	Op    string
	Left  Expression
	Right Expression
	Line  int
	Col   int
}

func (b *BinaryExpr) Pos() (int, int) { return b.Line, b.Col }
func (b *BinaryExpr) exprNode()       {}

type UnaryExpr struct {
	Op   string
	X    Expression
	Line int
	Col  int
}

func (u *UnaryExpr) Pos() (int, int) { return u.Line, u.Col }
func (u *UnaryExpr) exprNode()       {}

type CallExpr struct {
	Callee    Expression
	Arguments []Expression
	Line      int
	Col       int
}

func (c *CallExpr) Pos() (int, int) { return c.Line, c.Col }
func (c *CallExpr) exprNode()       {}

type MemberExpr struct {
	Object   Expression
	Property Expression
	Computed bool // obj[prop] vs obj.prop
	Line     int
	Col      int
}

func (m *MemberExpr) Pos() (int, int) { return m.Line, m.Col }
func (m *MemberExpr) exprNode()       {}

// ========== 语句节点 ==========

type ExpressionStmt struct {
	Expr Expression
}

func (e *ExpressionStmt) Pos() (int, int) { return e.Expr.Pos() }
func (e *ExpressionStmt) stmtNode()       {}

type VarDecl struct {
	Kind string // "var" | "let" | "const"
	Name Expression // Allow Identifier or destructuring pattern
	Init Expression // 可能为 nil
	Line int
	Col  int
}

func (v *VarDecl) Pos() (int, int) { return v.Line, v.Col }
func (v *VarDecl) stmtNode()       {}

type BlockStmt struct {
	Body []Statement
	Line int
	Col  int
}

func (b *BlockStmt) Pos() (int, int) { return b.Line, b.Col }
func (b *BlockStmt) stmtNode()       {}

type IfStmt struct {
	Test       Expression
	Consequent Statement
	Alternate  Statement // 可能为 nil
	Line       int
	Col        int
}

func (i *IfStmt) Pos() (int, int) { return i.Line, i.Col }
func (i *IfStmt) stmtNode()       {}

type ForStmt struct {
	Init   Statement  // 可能为 nil
	Test   Expression // 可能为 nil
	Update Expression // 可能为 nil
	Body   Statement
	Line   int
	Col    int
}

func (f *ForStmt) Pos() (int, int) { return f.Line, f.Col }
func (f *ForStmt) stmtNode()       {}

type WhileStmt struct {
	Test Expression
	Body Statement
	Line int
	Col  int
}

func (w *WhileStmt) Pos() (int, int) { return w.Line, w.Col }
func (w *WhileStmt) stmtNode()       {}

type ReturnStmt struct {
	Argument Expression // 可能为 nil
	Line     int
	Col      int
}

func (r *ReturnStmt) Pos() (int, int) { return r.Line, r.Col }
func (r *ReturnStmt) stmtNode()       {}

// 函数声明/表达式
type FunctionDecl struct {
	Name      *Identifier // 可能为 nil（匿名函数表达式）
	Params    []*Identifier
	Body      *BlockStmt
	Generator bool // 是否为生成器函数
	Async     bool // 是否为 async
	Line      int
	Col       int
}

func (f *FunctionDecl) Pos() (int, int) { return f.Line, f.Col }
func (f *FunctionDecl) stmtNode()       {}
func (f *FunctionDecl) exprNode()       {}

// Try-Catch 语句
type TryCatchStmt struct {
	TryBlock     *BlockStmt
	CatchParam   *Identifier // catch (e) 中的 e
	CatchBlock   *BlockStmt  // 可能为 nil
	FinallyBlock *BlockStmt  // 可能为 nil
	Line         int
	Col          int
}

func (t *TryCatchStmt) Pos() (int, int) { return t.Line, t.Col }
func (t *TryCatchStmt) stmtNode()       {}

// Throw 语句
type ThrowStmt struct {
	Argument Expression
	Line     int
	Col      int
}

func (t *ThrowStmt) Pos() (int, int) { return t.Line, t.Col }
func (t *ThrowStmt) stmtNode()       {}

// Break 语句
type BreakStmt struct {
	Line int
	Col  int
}

func (b *BreakStmt) Pos() (int, int) { return b.Line, b.Col }
func (b *BreakStmt) stmtNode()       {}

// Continue 语句
type ContinueStmt struct {
	Label *Identifier // 可能为 nil
	Line  int
	Col   int
}

func (c *ContinueStmt) Pos() (int, int) { return c.Line, c.Col }
func (c *ContinueStmt) stmtNode()       {}

// Switch 语句
type SwitchStmt struct {
	Discriminant Expression
	Cases        []*CaseClause
	Line, Col    int
}

func (s *SwitchStmt) Pos() (int, int) { return s.Line, s.Col }
func (s *SwitchStmt) stmtNode()       {}

type CaseClause struct {
	Test       Expression // nil 表示 default
	Consequent []Statement
	Line, Col  int
}

// Class 声明
type ClassDecl struct {
	Name       *Identifier
	SuperClass Expression // 可能为 nil
	Body       []*MethodDef
	Line, Col  int
}

func (c *ClassDecl) Pos() (int, int) { return c.Line, c.Col }
func (c *ClassDecl) stmtNode()       {}
func (c *ClassDecl) exprNode()       {}

type MethodDef struct {
	Key       *Identifier
	Value     *FunctionDecl
	Static    bool
	Kind      string // "constructor" | "method" | "get" | "set"
	Line, Col int
}

// New 表达式
type NewExpr struct {
	Callee    Expression
	Arguments []Expression
	Line, Col int
}

func (n *NewExpr) Pos() (int, int) { return n.Line, n.Col }
func (n *NewExpr) exprNode()       {}

// Delete 表达式
type DeleteExpr struct {
	Argument  Expression
	Line, Col int
}

func (d *DeleteExpr) Pos() (int, int) { return d.Line, d.Col }
func (d *DeleteExpr) exprNode()       {}

// Update 表达式（自增自减）
type UpdateExpr struct {
	Op        string // "++" 或 "--"
	Argument  Expression
	Prefix    bool
	Line, Col int
}

func (u *UpdateExpr) Pos() (int, int) { return u.Line, u.Col }
func (u *UpdateExpr) exprNode()       {}

// 展开元素（...）
type SpreadElement struct {
	Argument  Expression
	Line, Col int
}

func (s *SpreadElement) Pos() (int, int) { return s.Line, s.Col }
func (s *SpreadElement) exprNode()       {}

// 箭头函数表达式
type ArrowFunctionExpr struct {
	Params    []*Identifier
	Body      Node // BlockStmt 或 Expression
	Async     bool
	Line, Col int
}

func (a *ArrowFunctionExpr) Pos() (int, int) { return a.Line, a.Col }
func (a *ArrowFunctionExpr) exprNode()       {}

// 三元表达式 cond ? expr1 : expr2
type ConditionalExpr struct {
	Test       Expression
	Consequent Expression
	Alternate  Expression
	Line, Col  int
}

func (c *ConditionalExpr) Pos() (int, int) { return c.Line, c.Col }
func (c *ConditionalExpr) exprNode()       {}

// 数组字面量 [a, b, ...]
type ArrayLiteral struct {
	Elements  []Expression // 可包含 SpreadElement
	Line, Col int
}

func (a *ArrayLiteral) Pos() (int, int) { return a.Line, a.Col }
func (a *ArrayLiteral) exprNode()       {}

// 对象字面量 {a: 1, b: 2, ...}
type ObjectLiteral struct {
	Properties []*Property
	Line, Col  int
}

func (o *ObjectLiteral) Pos() (int, int) { return o.Line, o.Col }
func (o *ObjectLiteral) exprNode()       {}

type Property struct {
	Key       Expression // Identifier 或 Literal
	Value     Expression
	Kind      string // "init" | "get" | "set"
	Computed  bool
	Method    bool
	Shorthand bool
	Line, Col int
}

// 模板字符串 `hello ${name}!`
type TemplateLiteral struct {
	Quasis      []string     // 字符串静态部分（切片）
	Expressions []Expression // 动态插值部分
	Line, Col   int
}

func (t *TemplateLiteral) Pos() (int, int) { return t.Line, t.Col }
func (t *TemplateLiteral) exprNode()       {}

// 赋值表达式 a = b, a += b, ...
type AssignmentExpr struct {
	Operator  string     // "=", "+=", "-=", "*=", "/=", 等
	Left      Expression // 通常是 Identifier、MemberExpr 等
	Right     Expression
	Line, Col int
}

func (a *AssignmentExpr) Pos() (int, int) { return a.Line, a.Col }
func (a *AssignmentExpr) exprNode()       {}

// 逗号表达式 (a, b, c)
type SequenceExpr struct {
	Expressions []Expression
	Line, Col   int
}

func (s *SequenceExpr) Pos() (int, int) { return s.Line, s.Col }
func (s *SequenceExpr) exprNode()       {}

// this 关键字
type ThisExpr struct {
	Line, Col int
}

func (t *ThisExpr) Pos() (int, int) { return t.Line, t.Col }
func (t *ThisExpr) exprNode()       {}

// super 关键字
type SuperExpr struct {
	Line, Col int
}

func (s *SuperExpr) Pos() (int, int) { return s.Line, s.Col }
func (s *SuperExpr) exprNode()       {}
