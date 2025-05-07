package simplejs

import (
	"strconv"
)

// parsePrimary handles literals, identifiers, parentheses, and arrow functions.
func (p *Parser) parsePrimary() (Expression, error) {
	tok := p.peek()

	// Try arrow function: (...) => expr or x => expr
	if tok.Type == TokLParen || tok.Type == TokIdentifier {
		pos := p.save()
		var params []*Identifier
		if tok.Type == TokLParen {
			// parse '(' params ')'
			p.next()
			for p.peek().Type != TokRParen {
				idTok := p.peek()
				if idTok.Type != TokIdentifier {
					p.restore(pos)
					params = nil
					break
				}
				p.next()
				params = append(params, &Identifier{Name: idTok.Literal, Line: idTok.Line, Col: 0})
				if p.peek().Type == TokComma {
					p.next()
					continue
				}
				break
			}
			if params != nil {
				if p.peek().Type != TokRParen {
					p.restore(pos)
					params = nil
				} else {
					p.next() // consume ')'
				}
			}
		} else {
			// single identifier parameter
			p.next()
			params = []*Identifier{{Name: tok.Literal, Line: tok.Line, Col: 0}}
		}
		if params != nil && p.peek().Type == TokArrow {
			p.next() // consume '=>'
			var body Node
			if p.peek().Type == TokLBrace {
				block, err := p.parseBlock()
				if err != nil {
					return nil, err
				}
				body = block
			} else {
				expr, err := p.ParseExpression()
				if err != nil {
					return nil, err
				}
				body = expr
			}
			return &ArrowFunctionExpr{Params: params, Body: body, Async: false, Line: tok.Line, Col: 0}, nil
		}
		p.restore(pos)
	}

	// Default literal, identifier, or other primary
	switch tok.Type {
	case TokNumber:
		p.next()
		num, err := strconv.ParseFloat(tok.Literal, 64)
		if err != nil {
			return nil, p.errorf("invalid number: %s", tok.Literal)
		}
		return &Literal{Value: num, Line: tok.Line, Col: 0}, nil
	case TokString:
		p.next()
		return &Literal{Value: tok.Literal, Line: tok.Line, Col: 0}, nil
	case TokBool:
		p.next()
		b := tok.Literal == "true"
		return &Literal{Value: b, Line: tok.Line, Col: 0}, nil
	case TokNull:
		p.next()
		return &Literal{Value: nil, Line: tok.Line, Col: 0}, nil
	case TokUndefined:
		p.next()
		return &Literal{Value: struct{}{}, Line: tok.Line, Col: 0}, nil
	case TokFunction:
		return p.parseFunctionDecl(true)
	case TokNew:
		return p.parseNew()
	case TokLBracket:
		return p.parseArrayLiteral()
	case TokLBrace:
		return p.parseObjectLiteral()
	case TokIdentifier:
		p.next()
		return &Identifier{Name: tok.Literal, Line: tok.Line, Col: 0}, nil
	case TokSuper:
		p.next()
		return &SuperExpr{Line: tok.Line, Col: 0}, nil
	default:
		return nil, p.errorf("unexpected token %v in primary", tok.Type)
	}
}

// parseObjectLiteral parses { key: value, ... }
func (p *Parser) parseObjectLiteral() (Expression, error) {
	// parse object literal: {key: value, ...}
	start := p.peek()
	_, err := p.expect(TokLBrace)
	if err != nil {
		return nil, err
	}
	properties := []*Property{}
	for p.peek().Type != TokRBrace && p.peek().Type != TokEOF {
		// key
		var key Expression
		if p.peek().Type == TokIdentifier {
			tok := p.next()
			key = &Identifier{Name: tok.Literal, Line: tok.Line, Col: 0}
		} else if p.peek().Type == TokString {
			tok := p.next()
			key = &Literal{Value: tok.Literal, Line: tok.Line, Col: 0}
		} else {
			return nil, p.errorf("unexpected object literal key: %v", p.peek().Type)
		}
		// colon
		_, err := p.expect(TokColon)
		if err != nil {
			return nil, err
		}
		// value expression
		valExpr, err := p.ParseExpression()
		if err != nil {
			return nil, err
		}
		lineNum, colNum := key.Pos()
		properties = append(properties, &Property{Key: key, Value: valExpr, Kind: "init", Computed: false, Method: false, Shorthand: false, Line: lineNum, Col: colNum})
		if p.peek().Type == TokComma {
			p.next()
		}
	}
	_, err = p.expect(TokRBrace)
	if err != nil {
		return nil, err
	}
	return &ObjectLiteral{Properties: properties, Line: start.Line, Col: 0}, nil
}

// parseArrayLiteral parses [elem1, elem2, ...]
func (p *Parser) parseArrayLiteral() (Expression, error) {
	// parse array literal: [expr, ...]
	start := p.peek()
	_, err := p.expect(TokLBracket)
	if err != nil {
		return nil, err
	}
	elements := []Expression{}
	for p.peek().Type != TokRBracket && p.peek().Type != TokEOF {
		expr, err := p.ParseExpression()
		if err != nil {
			return nil, err
		}
		elements = append(elements, expr)
		if p.peek().Type == TokComma {
			p.next()
		}
	}
	_, err = p.expect(TokRBracket)
	if err != nil {
		return nil, err
	}
	return &ArrayLiteral{Elements: elements, Line: start.Line, Col: 0}, nil
}

// parseObjectPattern parses {a, b, c} for destructuring assignment
func (p *Parser) parseObjectPattern() (Expression, error) {
	start := p.peek()
	_, err := p.expect(TokLBrace)
	if err != nil {
		return nil, err
	}
	properties := []*Property{}
	for p.peek().Type != TokRBrace && p.peek().Type != TokEOF {
		tok := p.peek()
		if tok.Type == TokIdentifier {
			nameTok := p.next()
			key := &Identifier{Name: nameTok.Literal, Line: nameTok.Line, Col: 0}
			var valueExpr Expression = key
			if p.peek().Type == TokColon {
				p.next()
				valTok := p.next()
				if valTok.Type != TokIdentifier {
					return nil, p.errorf("expected identifier after colon in object pattern, got %v", valTok.Type)
				}
				valueExpr = &Identifier{Name: valTok.Literal, Line: valTok.Line, Col: 0}
			}
			lineNum, colNum := key.Pos()
			properties = append(properties, &Property{Key: key, Value: valueExpr, Line: lineNum, Col: colNum})
			if p.peek().Type == TokComma {
				p.next()
			}
		} else if tok.Type == TokComma {
			p.next()
			continue
		} else {
			return nil, p.errorf("object pattern expects identifier or comma, got %v", tok.Type)
		}
	}
	_, err = p.expect(TokRBrace)
	if err != nil {
		return nil, err
	}
	return &ObjectLiteral{Properties: properties, Line: start.Line, Col: 0}, nil
}

// parseArrayPattern parses [a, b, ...] for destructuring assignment
func (p *Parser) parseArrayPattern() (Expression, error) {
	start := p.peek()
	_, err := p.expect(TokLBracket)
	if err != nil {
		return nil, err
	}
	elements := []Expression{}
	for p.peek().Type != TokRBracket && p.peek().Type != TokEOF {
		tok := p.peek()
		if tok.Type == TokIdentifier {
			nameTok := p.next()
			elements = append(elements, &Identifier{Name: nameTok.Literal, Line: nameTok.Line, Col: 0})
		} else if tok.Type == TokComma {
			p.next()
			continue
		} else {
			return nil, p.errorf("array pattern expects identifier or comma, got %v", tok.Type)
		}
		if p.peek().Type == TokComma {
			p.next()
		}
	}
	_, err = p.expect(TokRBracket)
	if err != nil {
		return nil, err
	}
	return &ArrayLiteral{Elements: elements, Line: start.Line, Col: 0}, nil
}
