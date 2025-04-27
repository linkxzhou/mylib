package simplejs

// parseWhile parses while (cond) stmt, building AST
func (p *Parser) parseWhile() (Statement, error) {
	// consume 'while'
	tok := p.next()
	_, err := p.expect(TokLParen)
	if err != nil {
		return nil, err
	}
	// parse condition
	test, err := p.ParseExpression()
	if err != nil {
		return nil, err
	}
	_, err = p.expect(TokRParen)
	if err != nil {
		return nil, err
	}
	// parse body
	body, err := p.ParseStatement()
	if err != nil {
		return nil, err
	}
	return &WhileStmt{Test: test, Body: body, Line: tok.Line, Col: 0}, nil
}

// parseFor parses for (init; cond; post) stmt, building AST
func (p *Parser) parseFor() (Statement, error) {
	// consume 'for'
	tok := p.next()
	_, err := p.expect(TokLParen)
	if err != nil {
		return nil, err
	}
	// init
	var init Statement
	if p.peek().Type != TokSemicolon {
		if p.peek().Type == TokLet || p.peek().Type == TokConst {
			init, err = p.parseLetConst()
		} else if p.peek().Type == TokVar {
			// inline var declaration without consuming semicolon
			tokVar := p.next() // 'var'
			if p.peek().Type != TokIdentifier {
				return nil, p.errorf("expected identifier after var")
			}
			nameTok := p.next()
			var initExpr Expression
			if p.peek().Type == TokAssign {
				p.next()
				initExpr, err = p.ParseExpression()
				if err != nil {
					return nil, err
				}
			}
			init = &VarDecl{Kind: "var", Name: &Identifier{Name: nameTok.Literal, Line: nameTok.Line, Col: 0}, Init: initExpr, Line: tokVar.Line, Col: 0}
		} else {
			expr, err2 := p.ParseExpression()
			if err2 != nil {
				return nil, err2
			}
			init = &ExpressionStmt{Expr: expr}
		}
		if err != nil {
			return nil, err
		}
	}
	// consume ';' after init
	if p.peek().Type == TokSemicolon {
		p.next()
	}
	// condition
	var test Expression
	if p.peek().Type != TokSemicolon {
		test, err = p.ParseExpression()
		if err != nil {
			return nil, err
		}
	}
	_, err = p.expect(TokSemicolon)
	if err != nil {
		return nil, err
	}
	// update
	var update Expression
	if p.peek().Type != TokRParen {
		update, err = p.ParseExpression()
		if err != nil {
			return nil, err
		}
	}
	_, err = p.expect(TokRParen)
	if err != nil {
		return nil, err
	}
	// body
	body, err := p.ParseStatement()
	if err != nil {
		return nil, err
	}
	return &ForStmt{Init: init, Test: test, Update: update, Body: body, Line: tok.Line, Col: 0}, nil
}
