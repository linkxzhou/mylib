package simplejs

import (
	"fmt"
	"strconv"
)

// ParseStatement parses a statement (expression;).
func (p *Parser) ParseStatement() (JSValue, error) {
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
		return p.parseFunctionDecl(true)
	case TokReturn:
		p.debug("Parse return statement")
		p.next()
		val := Undefined()
		if p.peek().Type != TokSemicolon && p.peek().Type != TokRBrace && p.peek().Type != TokEOF {
			v, err := p.ParseExpression()
			if err != nil {
				return v, err
			}
			val = v
		}
		p.debug("Return value: %v", val.ToString())
		panic(ReturnPanic{Value: val})
	case TokThrow:
		p.debug("Parse throw statement")
		p.next() // consume throw keyword
		expr, err := p.ParseExpression()
		if err != nil {
			return expr, err
		}
		if p.peek().Type == TokSemicolon {
			p.next() // optional semicolon
		}
		p.debug("Throw exception: %v", expr.ToString())
		// throw exception
		return Undefined(), fmt.Errorf("Uncaught exception: %s", expr.ToString())
	case TokTry:
		p.debug("Parse try-catch statement")
		return p.parseTryCatch()
	case TokLet, TokConst:
		p.debug("Parse let/const statement")
		return p.parseLetConst()
	case TokVar:
		p.debug("Parse var statement")
		return p.parseVar()
	case TokClass:
		p.debug("Parse class statement")
		return p.parseClass()
	case TokNew:
		p.debug("Parse new statement")
		return p.parseNew()
	case TokBreak:
		p.debug("Parse break statement")
		p.next()
		panic(BreakPanic{})
	default:
		p.debug("Parse expression statement")
		// expression statement
		val, err := p.ParseExpression()
		if err != nil {
			return val, err
		}
		if p.peek().Type == TokSemicolon {
			p.next()
		}
		p.debug("Expression statement value: %v", val.ToString())
		return val, nil
	}
}

// parseLetConst handles let/const variable declarations
func (p *Parser) parseLetConst() (JSValue, error) {
	p.next() // consume let/const
	var lastVal JSValue = Undefined()
	for {
		tok := p.peek()
		if tok.Type == TokIdentifier {
			// let/const name = ...
			name := p.next().Literal
			var val JSValue = Undefined()
			if p.peek().Type == TokAssign {
				p.next()
				v, err := p.ParseExpression()
				if err != nil {
					return v, err
				}
				val = v
			}
			p.ctx.global.Set(name, val)
			lastVal = val
		} else if tok.Type == TokLBracket || tok.Type == TokLBrace {
			// let/const [a, b] = ... or {x, y} = ...
			var patternType = tok.Type
			// Parse pattern as pattern (not as literal)
			oldPattern := p.inPattern
			p.inPattern = true
			var patVal JSValue
			var err error
			if patternType == TokLBracket {
				patVal, err = p.parseArrayPattern()
			} else {
				patVal, err = p.parseObjectPattern()
			}
			p.inPattern = oldPattern
			if err != nil {
				return patVal, err
			}
			if p.peek().Type != TokAssign {
				return Undefined(), p.errorf("expected '=' after destructuring pattern")
			}
			p.next() // consume '='
			rhs, err := p.ParseExpression()
			if err != nil {
				return rhs, err
			}
			// Destructure assignment (array or object)
			if patternType == TokLBracket {
				// Array destructuring
				if !rhs.IsObject() {
					return Undefined(), p.errorf("right-hand side of array destructuring must be an array-like object")
				}
				arr := rhs.ToObject()
				patArr := patVal.ToObject()
				for k, v := range patArr {
					if _, err := strconv.Atoi(k); err == nil {
						if val, ok := arr[k]; ok {
							// pattern: identifier
							if v.Type == JSIdentifier {
								p.ctx.global.Set(v.String, val)
							}
						}
					}
				}
				lastVal = rhs
			} else if patternType == TokLBrace {
				// Object destructuring
				if !rhs.IsObject() {
					return Undefined(), p.errorf("right-hand side of object destructuring must be an object")
				}
				obj := rhs.ToObject()
				patObj := patVal.ToObject()
				for k, v := range patObj {
					if v.Type == JSIdentifier {
						if val, ok := obj[k]; ok {
							p.ctx.global.Set(v.String, val)
						} else {
							p.ctx.global.Set(v.String, Undefined())
						}
					}
				}
				lastVal = rhs
			}
		} else {
			return Undefined(), p.errorf("expected variable name or destructuring pattern after let/const")
		}

		if p.peek().Type != TokComma {
			break
		}
		p.next() // consume comma
	}
	return lastVal, nil
}

// parseVar handles var variable declarations (ES5)
func (p *Parser) parseVar() (JSValue, error) {
	p.next() // consume var
	var lastVal JSValue = Undefined()
	for {
		if p.peek().Type != TokIdentifier {
			return Undefined(), p.errorf("expected variable name after var")
		}
		name := p.next().Literal
		var val JSValue = Undefined()
		if p.peek().Type == TokAssign {
			p.next()
			v, err := p.ParseExpression()
			if err != nil {
				return v, err
			}
			val = v
			// 确保函数返回值被正确处理，特别是当返回值是另一个函数时
			if val.Type == JSFunction {
				// 如果是函数，确保它能被正确赋值给变量
				p.ctx.global.Set(name, val)
			} else {
				p.ctx.global.Set(name, val)
			}
		} else {
			p.ctx.global.Set(name, val)
		}
		lastVal = val
		if p.peek().Type != TokComma {
			break
		}
		p.next() // consume comma
	}
	return lastVal, nil
}

// parseBlock parses a block { ... }
func (p *Parser) parseBlock() (JSValue, error) {
	_, err := p.expect(TokLBrace)
	if err != nil {
		return Undefined(), err
	}
	oldScope := p.ctx.global
	p.ctx.global = NewScope(oldScope)
	var res JSValue = Undefined()
	defer func() { p.ctx.global = oldScope }()
	for p.peek().Type != TokRBrace && p.peek().Type != TokEOF {
		stmt, err := p.ParseStatement()
		if err != nil {
			return stmt, err
		}
		res = stmt
		// 跳过可能的分号，允许多个语句
		for p.peek().Type == TokSemicolon {
			p.next()
		}
	}
	_, err = p.expect(TokRBrace)
	return res, err
}

// parseIf parses if (cond) stmt [else stmt]
func (p *Parser) parseIf() (JSValue, error) {
	// consume 'if' and '('
	if _, err := p.expect(TokIf); err != nil {
		return Undefined(), err
	}
	if _, err := p.expect(TokLParen); err != nil {
		return Undefined(), err
	}
	condVal, err := p.ParseExpression()
	if err != nil {
		return Undefined(), err
	}
	if _, err := p.expect(TokRParen); err != nil {
		return Undefined(), err
	}
	if condVal.ToBool() {
		thenVal, err := p.ParseStatement()
		if err != nil {
			return thenVal, err
		}
		// skip else branch
		if p.peek().Type == TokElse {
			p.next()
			p.skipStatement()
		}
		return thenVal, nil
	} else {
		// skip then branch
		p.skipStatement()
		// else branch
		if p.peek().Type == TokElse {
			p.next()
			elseVal, err := p.ParseStatement()
			if err != nil {
				return elseVal, err
			}
			return elseVal, nil
		}
		return Undefined(), nil
	}
}

// parseTryCatch 解析 try-catch 语句
func (p *Parser) parseTryCatch() (JSValue, error) {
	_, err := p.expect(TokTry)
	if err != nil {
		return Undefined(), err
	}

	// 解析 try 块
	tryBlock, tryErr := p.ParseStatement()

	// 跳过分号和右大括号，确保能识别 catch
	for {
		t := p.peek().Type
		if t == TokSemicolon || t == TokRBrace {
			p.next()
		} else {
			break
		}
	}

	// 必须有 catch 块
	_, err = p.expect(TokCatch)
	if err != nil {
		return Undefined(), err
	}

	// 解析 catch 参数 (e)
	_, err = p.expect(TokLParen)
	if err != nil {
		return Undefined(), err
	}

	// 获取异常变量名，允许 catch() 匿名变量
	var exceptionVar string
	if p.peek().Type == TokIdentifier {
		exceptionVar = p.next().Literal
	} else if p.peek().Type == TokRParen {
		exceptionVar = ""
	} else {
		return Undefined(), p.errorf("expected identifier or ) in catch clause")
	}

	_, err = p.expect(TokRParen)
	if err != nil {
		return Undefined(), err
	}

	// 创建新的作用域，将异常信息绑定到变量（如果有变量名）
	oldScope := p.ctx.global
	p.ctx.global = NewScope(oldScope)

	// 如果 try 块有异常，将异常信息绑定到变量
	if exceptionVar != "" {
		if tryErr != nil {
			p.ctx.global.Set(exceptionVar, StringVal(tryErr.Error()))
		} else {
			p.ctx.global.Set(exceptionVar, Undefined())
		}
	}

	// 解析 catch 块
	catchBlock, catchErr := p.ParseStatement()

	// 恢复原来的作用域
	p.ctx.global = oldScope

	// 如果 catch 块有异常，返回该异常
	if catchErr != nil {
		return Undefined(), catchErr
	}

	// 如果 try 块没有异常，返回 try 块的结果，否则返回 catch 块的结果
	var result JSValue
	if tryErr == nil {
		result = tryBlock
	} else {
		result = catchBlock
	}

	// 解析 try-catch 块后面的语句（如果有）
	if p.pos < len(p.tokens) && p.tokens[p.pos].Type != TokEOF && p.tokens[p.pos].Type != TokRBrace {
		// 如果后面还有语句，解析并返回它们的结果
		nextStmt, err := p.ParseStatement()
		if err != nil {
			return Undefined(), err
		}
		return nextStmt, nil
	}

	return result, nil
}
