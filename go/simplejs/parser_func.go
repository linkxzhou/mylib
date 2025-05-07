package simplejs

// parseFunctionDecl 生成 FunctionDecl AST
func (p *Parser) parseFunctionDecl(requireFunction bool) (Expression, error) {
	if requireFunction {
		p.next() // consume 'function'
	}
	// 函数名可选（匿名函数表达式）
	var nameIdent *Identifier
	if p.peek().Type == TokIdentifier {
		tokName := p.next()
		nameIdent = &Identifier{Name: tokName.Literal, Line: tokName.Line, Col: 0}
	}
	_, err := p.expect(TokLParen)
	if err != nil {
		return nil, err
	}
	// 解析参数列表
	var params []*Identifier
	for p.peek().Type != TokRParen {
		if p.peek().Type != TokIdentifier {
			return nil, p.errorf("expected parameter name")
		}
		tokParam := p.next()
		params = append(params, &Identifier{Name: tokParam.Literal, Line: tokParam.Line, Col: 0})
		if p.peek().Type == TokComma {
			p.next()
		} else {
			break
		}
	}
	_, err = p.expect(TokRParen)
	if err != nil {
		return nil, err
	}
	// 解析函数体块
	stmt, err := p.parseBlock()
	if err != nil {
		return nil, err
	}
	block, ok := stmt.(*BlockStmt)
	if !ok {
		return nil, p.errorf("invalid function body")
	}
	// 生成 FunctionDecl AST
	fnDecl := &FunctionDecl{Name: nameIdent, Params: params, Body: block, Line: block.Line, Col: block.Col}
	// DEBUG LOG: Print function declaration name and params
	p.debug("[parseFunctionDecl] FunctionDecl: %s, params: %d", func() string {
		if fnDecl.Name != nil {
			return fnDecl.Name.Name
		} else {
			return "<anonymous>"
		}
	}(), len(fnDecl.Params))
	return fnDecl, nil
}

// parseNew 生成 NewExpr AST
func (p *Parser) parseNew() (Expression, error) {
	p.next() // consume 'new'

	tok := p.peek()
	if tok.Type != TokIdentifier {
		return nil, p.errorf("expected constructor name after new")
	}
	nameTok := p.next()
	callee := &Identifier{Name: nameTok.Literal, Line: nameTok.Line, Col: 0}

	_, err := p.expect(TokLParen)
	if err != nil {
		return nil, err
	}

	var args []Expression
	for p.peek().Type != TokRParen && p.peek().Type != TokEOF {
		arg, err := p.ParseExpression()
		if err != nil {
			return nil, err
		}
		args = append(args, arg)
		if p.peek().Type == TokComma {
			p.next()
		} else {
			break
		}
	}

	_, err = p.expect(TokRParen)
	if err != nil {
		return nil, err
	}

	// 使用默认列号0，Token无Col字段
	return &NewExpr{Callee: callee, Arguments: args, Line: tok.Line, Col: 0}, nil
}
