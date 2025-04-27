package simplejs

import (
	"fmt"
	"strings"
)

// ParseException is an error during parsing.
type ParseException struct {
	Msg    string
	Line   int
	Column int
	Code   string
}

func (e ParseException) Error() string {
	return fmt.Sprintf("Parse error at line %d, column %d: %s\nCode: %s", e.Line, e.Column, e.Msg, e.Code)
}

// Parser holds tokens and context.
type Parser struct {
	tokens      []Token
	pos         int
	ctx         *RunContext
	debugString []string
}

// NewParser creates a parser with tokens and runtime context.
func NewParser(tokens []Token, ctx *RunContext) *Parser {
	return &Parser{tokens: tokens, pos: 0, ctx: ctx}
}

// ParseExpression parses an expression and returns its AST node.
func (p *Parser) ParseExpression() (Expression, error) {
	return p.parseAssignment()
}

// ParseProgram parses a sequence of statements until EOF and builds an AST.
func (p *Parser) ParseProgram() (*Program, error) {
	prog := &Program{}
	for p.peek().Type != TokEOF {
		stmt, err := p.ParseStatement()
		if err != nil {
			return nil, err
		}
		prog.Body = append(prog.Body, stmt)
		// 跳过多余的分号
		for p.peek().Type == TokSemicolon {
			p.next()
		}
	}
	return prog, nil
}

// ReturnPanic is used to unwind stack for return statements
type ReturnPanic struct {
	Value JSValue
}

// errorf creates a ParseException with the current line information
func (p *Parser) errorf(format string, args ...interface{}) error {
	line, col, code := p.currentLineInfo()
	return ParseException{Msg: fmt.Sprintf(format, args...), Line: line, Column: col, Code: code}
}

// debug adds a debug message to the parser
func (p *Parser) debug(format string, args ...interface{}) {
	p.debugString = append(p.debugString, fmt.Sprintf(format, args...))
}

// 获取当前Token所在的行号和原始代码
func (p *Parser) currentLineInfo() (int, int, string) {
	if p.pos < len(p.tokens) {
		tok := p.tokens[p.pos]
		lineNum := tok.Line
		var codeLine string
		if ctx := p.ctx; ctx != nil && ctx.sourceLines != nil && lineNum > 0 {
			lines := *ctx.sourceLines
			if lineNum-1 < len(lines) {
				codeLine = lines[lineNum-1]
			}
		}
		col := 1
		if tok.Literal != "" && codeLine != "" {
			if idx := strings.Index(codeLine, tok.Literal); idx >= 0 {
				col = idx + 1
			}
		}
		return lineNum, col, codeLine
	}
	return 0, 0, ""
}

// peek returns current token.
func (p *Parser) peek() Token {
	if p.pos < len(p.tokens) {
		return p.tokens[p.pos]
	}
	return Token{Type: TokEOF}
}

// next returns current token and advances.
func (p *Parser) next() Token {
	tok := p.peek()
	p.pos++
	return tok
}

// expect consumes expected token or error.
func (p *Parser) expect(tt TokenType) (Token, error) {
	tok := p.next()
	if tok.Type != tt {
		return tok, p.errorf("expected %v, got %v (%q)", tt, tok.Type, tok.Literal)
	}
	return tok, nil
}
