package simplejs

// 解析阶段只产出 AST，当真实执行交给 interpreter

// ParseStatement parses a statement (expression;).
func (p *Parser) ParseStatement() (Statement, error) {
	tok := p.peek()
	p.debug("ParseStatement: token=%v[%s]", tok.Type, tok.Type.String())
	switch tok.Type {
	case TokLBrace:
		p.debug("Parse block statement")
		return p.parseBlock()
	case TokIf:
		p.debug("Parse if statement")
		return p.parseIf()
	case TokWhile:
		p.debug("Parse while statement")
		return p.parseWhile()
	case TokFor:
		p.debug("Parse for statement")
		return p.parseFor()
	case TokFunction:
		p.debug("Parse function declaration")
		// 将函数声明解析值包装为表达式语句，以满足 Statement 接口
		val, err := p.parseFunctionDecl(true)
		if err != nil {
			return nil, err
		}
		return &ExpressionStmt{Expr: val}, nil
	case TokReturn:
		p.debug("Parse return statement")
		return p.parseReturn()
	case TokThrow:
		p.debug("Parse throw statement")
		return p.parseThrow()
	case TokTry:
		p.debug("Parse try-catch statement")
		return p.parseTryCatch()
	case TokLet, TokConst:
		// 变量声明 AST
		return p.parseLetConst()
	case TokVar:
		// parseVar 产出 VarDecl AST
		return p.parseVar()
	case TokClass:
		p.debug("Parse class statement")
		return p.parseClass()
	case TokNew:
		// new 表达式作为表达式语句
		expr, err := p.parseNew()
		if err != nil {
			return nil, err
		}
		if p.peek().Type == TokSemicolon {
			p.next()
		}
		return &ExpressionStmt{Expr: expr}, nil
	case TokBreak:
		p.debug("Parse break statement")
		return p.parseBreak()
	default:
		p.debug("Parse expression statement")
		expr, err := p.ParseExpression()
		if err != nil {
			return nil, err
		}
		if p.peek().Type == TokSemicolon {
			p.next()
		}
		return &ExpressionStmt{Expr: expr}, nil
	}
}

// parseLetConst 产出 VarDecl AST，可支持多重声明
func (p *Parser) parseLetConst() (Statement, error) {
	tok := p.next() // let 或 const
	kind := tok.Literal
	var decls []Statement
	for {
		peekTok := p.peek()
		if peekTok.Type == TokIdentifier {
			nameTok := p.next()
			var init Expression
			if p.peek().Type == TokAssign {
				p.next()
				e, err := p.ParseExpression()
				if err != nil {
					return nil, err
				}
				init = e
			}
			decls = append(decls, &VarDecl{Kind: kind, Name: &Identifier{Name: nameTok.Literal, Line: nameTok.Line, Col: 0}, Init: init, Line: nameTok.Line, Col: 0})
		} else if peekTok.Type == TokLBrace {
			pattern, err := p.parseObjectPattern()
			if err != nil {
				return nil, err
			}
			var init Expression
			if p.peek().Type == TokAssign {
				p.next()
				e, err := p.ParseExpression()
				if err != nil {
					return nil, err
				}
				init = e
			}
			decls = append(decls, &VarDecl{Kind: kind, Name: pattern, Init: init, Line: peekTok.Line, Col: 0})
		} else if peekTok.Type == TokLBracket {
			pattern, err := p.parseArrayPattern()
			if err != nil {
				return nil, err
			}
			var init Expression
			if p.peek().Type == TokAssign {
				p.next()
				e, err := p.ParseExpression()
				if err != nil {
					return nil, err
				}
				init = e
			}
			decls = append(decls, &VarDecl{Kind: kind, Name: pattern, Init: init, Line: peekTok.Line, Col: 0})
		} else {
			return nil, p.errorf("expected identifier or pattern after %s", kind)
		}
		if p.peek().Type != TokComma {
			break
		}
		p.next()
	}
	if len(decls) == 1 {
		return decls[0], nil
	}
	return &BlockStmt{Body: decls, Line: tok.Line, Col: 0}, nil
}

// parseVar 产出 VarDecl AST
func (p *Parser) parseVar() (Statement, error) {
	tok := p.next() // var
	var decls []Statement
	for {
		peekTok := p.peek()
		if peekTok.Type == TokIdentifier {
			nameTok := p.next()
			var init Expression
			if p.peek().Type == TokAssign {
				p.next()
				e, err := p.ParseExpression()
				if err != nil {
					return nil, err
				}
				init = e
			}
			decls = append(decls, &VarDecl{Kind: "var", Name: &Identifier{Name: nameTok.Literal, Line: nameTok.Line, Col: 0}, Init: init, Line: nameTok.Line, Col: 0})
		} else if peekTok.Type == TokLBrace {
			pattern, err := p.parseObjectPattern()
			if err != nil {
				return nil, err
			}
			var init Expression
			if p.peek().Type == TokAssign {
				p.next()
				e, err := p.ParseExpression()
				if err != nil {
					return nil, err
				}
				init = e
			}
			decls = append(decls, &VarDecl{Kind: "var", Name: pattern, Init: init, Line: peekTok.Line, Col: 0})
		} else if peekTok.Type == TokLBracket {
			pattern, err := p.parseArrayPattern()
			if err != nil {
				return nil, err
			}
			var init Expression
			if p.peek().Type == TokAssign {
				p.next()
				e, err := p.ParseExpression()
				if err != nil {
					return nil, err
				}
				init = e
			}
			decls = append(decls, &VarDecl{Kind: "var", Name: pattern, Init: init, Line: peekTok.Line, Col: 0})
		} else {
			return nil, p.errorf("expected identifier or pattern after var")
		}
		if p.peek().Type != TokComma {
			break
		}
		p.next()
	}
	if p.peek().Type == TokSemicolon {
		p.next()
	}
	if len(decls) == 1 {
		return decls[0], nil
	}
	return &BlockStmt{Body: decls, Line: tok.Line, Col: 0}, nil
}

// parseBlock parses a block { ... }
func (p *Parser) parseBlock() (Statement, error) {
	// consume '{' and set up new scope
	tok, err := p.expect(TokLBrace)
	if err != nil {
		return nil, err
	}
	oldScope := p.ctx.global
	p.ctx.global = NewScope(oldScope)
	defer func() { p.ctx.global = oldScope }()
	// collect statements
	stmts := []Statement{}
	for p.peek().Type != TokRBrace && p.peek().Type != TokEOF {
		stmt, err := p.ParseStatement()
		if err != nil {
			return nil, err
		}
		stmts = append(stmts, stmt)
		// skip semicolons
		for p.peek().Type == TokSemicolon {
			p.next()
		}
	}
	// consume '}' and return block
	_, err = p.expect(TokRBrace)
	return &BlockStmt{Body: stmts, Line: tok.Line, Col: 0}, err
}

// parseIf parses if (cond) stmt [else stmt]
func (p *Parser) parseIf() (Statement, error) {
	// 'if' '(' Test ')' Consequent [ 'else' Alternate ]
	tok, err := p.expect(TokIf)
	if err != nil {
		return nil, err
	}
	_, err = p.expect(TokLParen)
	if err != nil {
		return nil, err
	}
	test, err := p.ParseExpression()
	if err != nil {
		return nil, err
	}
	_, err = p.expect(TokRParen)
	if err != nil {
		return nil, err
	}
	cons, err := p.ParseStatement()
	if err != nil {
		return nil, err
	}
	var alt Statement
	if p.peek().Type == TokElse {
		p.next()
		alt, err = p.ParseStatement()
		if err != nil {
			return nil, err
		}
	}
	return &IfStmt{Test: test, Consequent: cons, Alternate: alt, Line: tok.Line, Col: 0}, nil
}

// parseTryCatch 解析 try-catch 语句
func (p *Parser) parseTryCatch() (Statement, error) {
	_, err := p.expect(TokTry)
	if err != nil {
		return nil, err
	}

	// TryBlock
	tryBlockStmt, err := p.ParseStatement()
	if err != nil {
		return nil, err
	}
	tryBlock, ok := tryBlockStmt.(*BlockStmt)
	if !ok {
		line, _ := tryBlockStmt.Pos()
		tryBlock = &BlockStmt{Body: []Statement{tryBlockStmt}, Line: line, Col: 0}
	}

	// catch
	_, err = p.expect(TokCatch)
	if err != nil {
		return nil, err
	}
	_, err = p.expect(TokLParen)
	if err != nil {
		return nil, err
	}
	idTok, err := p.expect(TokIdentifier)
	if err != nil {
		return nil, err
	}
	_, err = p.expect(TokRParen)
	if err != nil {
		return nil, err
	}
	catchStmt, err := p.ParseStatement()
	if err != nil {
		return nil, err
	}
	catchBlock, ok := catchStmt.(*BlockStmt)
	if !ok {
		catchBlock = &BlockStmt{Body: []Statement{catchStmt}, Line: idTok.Line, Col: 0}
	}
	return &TryCatchStmt{TryBlock: tryBlock, CatchParam: &Identifier{Name: idTok.Literal, Line: idTok.Line, Col: 0}, CatchBlock: catchBlock, FinallyBlock: nil, Line: idTok.Line, Col: 0}, nil
}

// parseReturn 仅生成 AST
func (p *Parser) parseReturn() (Statement, error) {
	tok, err := p.expect(TokReturn)
	if err != nil {
		return nil, err
	}
	var arg Expression
	if p.peek().Type != TokSemicolon {
		arg, err = p.ParseExpression()
		if err != nil {
			return nil, err
		}
	}
	if p.peek().Type == TokSemicolon {
		p.next()
	}
	return &ReturnStmt{Argument: arg, Line: tok.Line, Col: 0}, nil
}

// parseThrow 仅生成 AST
func (p *Parser) parseThrow() (Statement, error) {
	tok, err := p.expect(TokThrow)
	if err != nil {
		return nil, err
	}
	expr, err := p.ParseExpression()
	if err != nil {
		return nil, err
	}
	if p.peek().Type == TokSemicolon {
		p.next()
	}
	return &ThrowStmt{Argument: expr, Line: tok.Line, Col: 0}, nil
}

// parseClass 仅生成 AST
func (p *Parser) parseClass() (Statement, error) {
	// class Name [extends Super] { methods }
	_, err := p.expect(TokClass)
	if err != nil {
		return nil, err
	}
	idTok, err := p.expect(TokIdentifier)
	if err != nil {
		return nil, err
	}
	// optional extends
	var super Expression
	if p.peek().Type == TokExtends {
		p.next()
		super, err = p.ParseExpression()
		if err != nil {
			return nil, err
		}
	}
	// class body
	_, err = p.expect(TokLBrace)
	if err != nil {
		return nil, err
	}
	methods := []*MethodDef{}
	for p.peek().Type != TokRBrace && p.peek().Type != TokEOF {
		// method name
		keyTok := p.peek()
		if keyTok.Type != TokIdentifier {
			return nil, p.errorf("expected method name, got %v", keyTok.Type)
		}
		p.next()
		// parameters
		_, err = p.expect(TokLParen)
		if err != nil {
			return nil, err
		}
		params := []*Identifier{}
		for p.peek().Type != TokRParen {
			paramTok, err := p.expect(TokIdentifier)
			if err != nil {
				return nil, err
			}
			params = append(params, &Identifier{Name: paramTok.Literal, Line: paramTok.Line, Col: 0})
			if p.peek().Type == TokComma {
				p.next()
			}
		}
		_, err = p.expect(TokRParen)
		if err != nil {
			return nil, err
		}
		// method body
		bodyStmt, err := p.parseBlock()
		if err != nil {
			return nil, err
		}
		block, ok := bodyStmt.(*BlockStmt)
		if !ok {
			return nil, p.errorf("method body is not block")
		}
		funcDecl := &FunctionDecl{Name: nil, Params: params, Body: block, Line: keyTok.Line, Col: 0}
		methods = append(methods, &MethodDef{Key: &Identifier{Name: keyTok.Literal, Line: keyTok.Line, Col: 0}, Value: funcDecl, Static: false, Kind: keyTok.Literal, Line: keyTok.Line, Col: 0})
	}
	_, err = p.expect(TokRBrace)
	if err != nil {
		return nil, err
	}
	return &ClassDecl{Name: &Identifier{Name: idTok.Literal, Line: idTok.Line, Col: 0}, SuperClass: super, Body: methods, Line: idTok.Line, Col: 0}, nil
}

// parseBreak 仅生成 AST
func (p *Parser) parseBreak() (Statement, error) {
	tok, err := p.expect(TokBreak)
	if err != nil {
		return nil, err
	}
	if p.peek().Type == TokSemicolon {
		p.next()
	}
	return &BreakStmt{Line: tok.Line, Col: 0}, nil
}
