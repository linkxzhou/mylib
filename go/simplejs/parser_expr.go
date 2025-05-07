package simplejs

// parseAssignment handles identifier = expr (recursive, right-associative)
func (p *Parser) parseAssignment() (Expression, error) {
	left, err := p.parseTernary()
	if err != nil {
		return nil, err
	}
	switch p.peek().Type {
	case TokAssign, TokPlusAssign, TokMinusAssign, TokAsteriskAssign, TokSlashAssign:
		tok := p.next()
		right, err := p.parseAssignment()
		if err != nil {
			return nil, err
		}
		return &AssignmentExpr{Operator: tok.Type.String(), Left: left, Right: right, Line: tok.Line, Col: 0}, nil
	}
	return left, nil
}

// parseLogicalOr handles ||
func (p *Parser) parseLogicalOr() (Expression, error) {
	left, err := p.parseLogicalAnd()
	if err != nil {
		return nil, err
	}
	for p.peek().Type == TokLogicalOr {
		tok := p.next()
		right, err := p.parseLogicalAnd()
		if err != nil {
			return nil, err
		}
		left = &BinaryExpr{Op: tok.Type.String(), Left: left, Right: right, Line: tok.Line, Col: 0}
	}
	return left, nil
}

// parseLogicalAnd handles &&
func (p *Parser) parseLogicalAnd() (Expression, error) {
	left, err := p.parseEquality()
	if err != nil {
		return nil, err
	}
	for p.peek().Type == TokLogicalAnd {
		tok := p.next()
		right, err := p.parseEquality()
		if err != nil {
			return nil, err
		}
		left = &BinaryExpr{Op: tok.Type.String(), Left: left, Right: right, Line: tok.Line, Col: 0}
	}
	return left, nil
}

// parseEquality handles ==, !=, ===, !==
func (p *Parser) parseEquality() (Expression, error) {
	left, err := p.parseRelational()
	if err != nil {
		return nil, err
	}
	for p.peek().Type == TokEqual || p.peek().Type == TokNotEqual || p.peek().Type == TokStrictEqual || p.peek().Type == TokNotStrictEqual {
		tok := p.next()
		right, err := p.parseRelational()
		if err != nil {
			return nil, err
		}
		left = &BinaryExpr{Op: tok.Type.String(), Left: left, Right: right, Line: tok.Line, Col: 0}
	}
	return left, nil
}

// parseRelational handles <, >, <=, >=
func (p *Parser) parseRelational() (Expression, error) {
	left, err := p.parseAddSub()
	if err != nil {
		return nil, err
	}
	for p.peek().Type == TokLT || p.peek().Type == TokGT || p.peek().Type == TokLTE || p.peek().Type == TokGTE {
		tok := p.next()
		right, err := p.parseAddSub()
		if err != nil {
			return nil, err
		}
		left = &BinaryExpr{Op: tok.Type.String(), Left: left, Right: right, Line: tok.Line, Col: 0}
	}
	return left, nil
}

// parseUnary handles unary operators like !, -, delete
func (p *Parser) parseUnary() (Expression, error) {
	tok := p.peek()
	switch tok.Type {
	case TokLogicalNot, TokMinus:
		tok := p.next()
		expr, err := p.parseUnary()
		if err != nil {
			return nil, err
		}
		return &UnaryExpr{Op: tok.Type.String(), X: expr, Line: tok.Line, Col: 0}, nil
	case TokDelete:
		tok := p.next()
		expr, err := p.parseUnary()
		if err != nil {
			return nil, err
		}
		return &DeleteExpr{Argument: expr, Line: tok.Line, Col: 0}, nil
	default:
		return p.parsePrimary()
	}
}

// parsePostfix handles postfix ++, -- and function calls
func (p *Parser) parsePostfix() (Expression, error) {
	left, err := p.parseUnary()
	if err != nil {
		return nil, err
	}
	// support postfix ++, -- and function calls
	for {
		switch p.peek().Type {
		case TokInc, TokDec:
			tok := p.next()
			left = &UpdateExpr{Op: tok.Type.String(), Argument: left, Prefix: false, Line: tok.Line, Col: 0}
		case TokDot:
			// member access obj.prop
			p.next()
			propTok, err := p.expect(TokIdentifier)
			if err != nil {
				return nil, err
			}
			left = &MemberExpr{Object: left, Property: &Identifier{Name: propTok.Literal, Line: propTok.Line, Col: 0}, Computed: false, Line: propTok.Line, Col: 0}
			continue
		case TokLBracket:
			// computed member access obj[expr]
			lbTok := p.next()
			expr, err := p.ParseExpression()
			if err != nil {
				return nil, err
			}
			if _, err := p.expect(TokRBracket); err != nil {
				return nil, err
			}
			left = &MemberExpr{Object: left, Property: expr, Computed: true, Line: lbTok.Line, Col: 0}
			continue
		case TokLParen:
			// parse call expression
			tok := p.next()
			var args []Expression
			if p.peek().Type != TokRParen {
				for {
					arg, err := p.ParseExpression()
					if err != nil {
						return nil, err
					}
					args = append(args, arg)
					if p.peek().Type != TokComma {
						break
					}
					p.next()
				}
			}
			if _, err := p.expect(TokRParen); err != nil {
				return nil, err
			}
			left = &CallExpr{Callee: left, Arguments: args, Line: tok.Line, Col: 0}
		default:
			return left, nil
		}
	}
}

// parseAddSub handles + and -.
func (p *Parser) parseAddSub() (Expression, error) {
	left, err := p.parseMulDiv()
	if err != nil {
		return nil, err
	}
	for p.peek().Type == TokPlus || p.peek().Type == TokMinus {
		tok := p.next()
		right, err := p.parseMulDiv()
		if err != nil {
			return nil, err
		}
		left = &BinaryExpr{Op: tok.Type.String(), Left: left, Right: right, Line: tok.Line, Col: 0}
	}
	return left, nil
}

// parseMulDiv handles * and /.
func (p *Parser) parseMulDiv() (Expression, error) {
	left, err := p.parsePostfix()
	if err != nil {
		return nil, err
	}
	for p.peek().Type == TokAsterisk || p.peek().Type == TokSlash {
		tok := p.next()
		right, err := p.parsePostfix()
		if err != nil {
			return nil, err
		}
		left = &BinaryExpr{Op: tok.Type.String(), Left: left, Right: right, Line: tok.Line, Col: 0}
	}
	return left, nil
}

// parseTernary handles ternary operator (cond ? expr1 : expr2)
func (p *Parser) parseTernary() (Expression, error) {
	left, err := p.parseLogicalOr()
	if err != nil {
		return nil, err
	}
	if p.peek().Type == TokQuestion {
		tok := p.next()
		cons, err := p.parseAssignment()
		if err != nil {
			return nil, err
		}
		if p.peek().Type != TokColon {
			return nil, p.errorf("expected ':' in conditional expression")
		}
		p.next()
		alt, err := p.parseAssignment()
		if err != nil {
			return nil, err
		}
		return &ConditionalExpr{Test: left, Consequent: cons, Alternate: alt, Line: tok.Line, Col: 0}, nil
	}
	return left, nil
}
