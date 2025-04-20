package simplejs

// parseWhile parses while (cond) stmt
func (p *Parser) parseWhile() (JSValue, error) {
	_, err := p.expect(TokWhile)
	if err != nil {
		return Undefined(), err
	}
	_, err = p.expect(TokLParen)
	if err != nil {
		return Undefined(), err
	}

	// 保存条件表达式的开始位置
	condStart := p.pos

	// 解析条件表达式
	_, err = p.ParseExpression()
	if err != nil {
		return Undefined(), err
	}

	_, err = p.expect(TokRParen)
	if err != nil {
		return Undefined(), err
	}

	// 保存循环体的开始和结束位置
	bodyStart := p.pos
	bodyEnd := 0
	var res JSValue = Undefined()
	breakOuter := false
	for {
		// 重新解析条件表达式
		p.pos = condStart
		cond, err := p.ParseExpression()
		if err != nil {
			return Undefined(), err
		}
		p.debug("While cond: %v", cond.ToString())
		if !cond.ToBool() {
			break
		}
		_, err = p.expect(TokRParen)
		if err != nil {
			return Undefined(), err
		}
		p.pos = bodyStart
		// 执行循环体，捕获 break
		func() {
			defer func() {
				if r := recover(); r != nil {
					if _, ok := r.(BreakPanic); ok {
						breakOuter = true
					} else {
						panic(r)
					}
				}
			}()
			res, err = p.ParseStatement()
		}()
		if breakOuter {
			break
		}
		if err != nil {
			return Undefined(), err
		}
		p.debug("While body result: %v", res.ToString())
		if bodyEnd == 0 {
			bodyEnd = p.pos
		} else {
			p.pos = bodyEnd
		}
	}
	if bodyEnd != 0 {
		p.pos = bodyEnd
	} else {
		_, err = p.ParseStatement()
		if err != nil {
			return Undefined(), err
		}
	}
	return res, nil
}

// parseFor parses for (init; cond; post) stmt
func (p *Parser) parseFor() (JSValue, error) {
	var err error
	// consume 'for' and '('
	if _, err = p.expect(TokFor); err != nil {
		return Undefined(), err
	}
	if _, err = p.expect(TokLParen); err != nil {
		return Undefined(), err
	}
	// new scope for init
	old := p.ctx.global
	p.ctx.global = NewScope(old)
	defer func() { p.ctx.global = old }()
	// init part
	if p.peek().Type != TokSemicolon {
		if p.peek().Type == TokLet || p.peek().Type == TokConst {
			if _, err = p.parseLetConst(); err != nil {
				return Undefined(), err
			}
		} else if p.peek().Type == TokVar {
			if _, err = p.parseVar(); err != nil {
				return Undefined(), err
			}
		} else {
			if _, err = p.ParseExpression(); err != nil {
				return Undefined(), err
			}
		}
	}
	// semicolon after init
	if _, err = p.expect(TokSemicolon); err != nil {
		return Undefined(), err
	}
	// record condition position
	condPos := p.pos
	var condVal JSValue = BoolVal(true)
	if p.peek().Type != TokSemicolon {
		if condVal, err = p.ParseExpression(); err != nil {
			return Undefined(), err
		}
	}
	// semicolon after condition
	if _, err = p.expect(TokSemicolon); err != nil {
		return Undefined(), err
	}
	// capture update tokens (do not execute now)
	var updateTokens []Token
	for p.peek().Type != TokRParen && p.peek().Type != TokEOF {
		updateTokens = append(updateTokens, p.next())
	}
	// consume ')'
	if _, err = p.expect(TokRParen); err != nil {
		return Undefined(), err
	}
	// loop body start
	bodyPos := p.pos
	var lastVal JSValue = Undefined()
	var breakLoop bool
	for {
		// evaluate condition
		p.pos = condPos
		if p.peek().Type != TokSemicolon {
			if condVal, err = p.ParseExpression(); err != nil {
				return Undefined(), err
			}
		}
		if !condVal.ToBool() {
			break
		}
		// execute body with break handling
		p.pos = bodyPos
		var val JSValue
		func() {
			defer func() {
				if r := recover(); r != nil {
					if _, ok := r.(BreakPanic); ok {
						breakLoop = true
						return
					}
					panic(r)
				}
			}()
			val, err = p.ParseStatement()
		}()
		if err != nil {
			return Undefined(), err
		}
		lastVal = val
		if breakLoop {
			break
		}
		// perform update exactly once
		if len(updateTokens) > 0 {
			sub := NewParser(updateTokens, p.ctx)
			if _, err = sub.ParseExpression(); err != nil {
				return Undefined(), err
			}
		}
	}
	return lastVal, nil
}
