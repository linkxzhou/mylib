package simplejs

import (
	"strconv"
)

// parsePrimary handles literals, identifiers, and parentheses.
func (p *Parser) parsePrimary() (Expression, error) {
	tok := p.peek()
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
	case TokFunction:
		// function expression or declaration without statement context
		return p.parseFunctionDecl(true)
	case TokNew:
		// new expression
		return p.parseNew()
	case TokLBracket:
		// array literal
		return p.parseArrayLiteral()
	case TokLBrace:
		// object literal
		return p.parseObjectLiteral()
	default:
		p.next()
		return &Identifier{Name: tok.Literal, Line: tok.Line, Col: 0}, nil
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
		} else {
			return nil, p.errorf("object pattern expects identifier or comma, got %v", p.peek().Type)
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
			p.next() // skip comma
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
