package simplejs

import (
	"fmt"
)

// ParseException is an error during parsing.
type ParseException struct {
	Msg  string
	Line int
	Code string
}

func (e ParseException) Error() string {
	return fmt.Sprintf("Parse error at line %d: %s\nCode: %s", e.Line, e.Msg, e.Code)
}

// Parser holds tokens and context.
type Parser struct {
	tokens      []Token
	pos         int
	ctx         *RunContext
	debugString []string
	inPattern   bool // true when parsing destructuring pattern
}

// NewParser creates a parser with tokens and runtime context.
func NewParser(tokens []Token, ctx *RunContext) *Parser {
	return &Parser{tokens: tokens, pos: 0, ctx: ctx}
}

// ParseExpression parses an expression.
func (p *Parser) ParseExpression() (JSValue, error) {
	return p.parseAssignment()
}

// ParseProgram parses a sequence of statements until EOF
func (p *Parser) ParseProgram() (JSValue, error) {
	var res JSValue = Undefined()
	for p.peek().Type != TokEOF {
		stmt, err := p.ParseStatement()
		if err != nil {
			return stmt, err
		}
		res = stmt
		// 跳过多余的分号
		for p.peek().Type == TokSemicolon {
			p.next()
		}
	}
	return res, nil
}

// ReturnPanic is used to unwind stack for return statements
type ReturnPanic struct {
	Value JSValue
}

// errorf creates a ParseException with the current line information
func (p *Parser) errorf(format string, args ...interface{}) error {
	line, code := p.currentLineInfo()
	return ParseException{Msg: fmt.Sprintf(format, args...), Line: line, Code: code}
}

// debug adds a debug message to the parser
func (p *Parser) debug(format string, args ...interface{}) {
	p.debugString = append(p.debugString, fmt.Sprintf(format, args...))
}

// 获取当前Token所在的行号和原始代码
func (p *Parser) currentLineInfo() (int, string) {
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
		return lineNum, codeLine
	}
	return 0, ""
}

// skipStatement skips a statement or block without evaluating semantics.
func (p *Parser) skipStatement() {
	tok := p.peek()
	switch tok.Type {
	case TokLBrace:
		p.next()
		depth := 1
		for depth > 0 && p.peek().Type != TokEOF {
			if p.peek().Type == TokLBrace {
				depth++
			} else if p.peek().Type == TokRBrace {
				depth--
			}
			p.next()
		}
	default:
		for p.peek().Type != TokSemicolon && p.peek().Type != TokRBrace && p.peek().Type != TokEOF {
			if p.peek().Type == TokLParen {
				p.next()
				depth := 1
				for depth > 0 && p.peek().Type != TokEOF {
					if p.peek().Type == TokLParen {
						depth++
					} else if p.peek().Type == TokRParen {
						depth--
					}
					p.next()
				}
			} else {
				p.next()
			}
		}
		if p.peek().Type == TokSemicolon {
			p.next()
		}
	}
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
