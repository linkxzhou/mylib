package simplejs

import (
	"strconv"
)

// parsePrimary handles literals, identifiers, and parentheses.
func (p *Parser) parsePrimary() (JSValue, error) {
	tok := p.peek()
	switch tok.Type {
	case TokNumber:
		p.next()
		num, err := strconv.ParseFloat(tok.Literal, 64)
		if err != nil {
			return Undefined(), p.errorf("invalid number: %s", tok.Literal)
		}
		return NumberVal(num), nil
	case TokString:
		p.next()
		return StringVal(tok.Literal), nil
	case TokBool:
		p.next()
		return BoolVal(tok.Literal == "true"), nil
	case TokNull:
		p.next()
		return Null(), nil
	case TokUndefined:
		p.next()
		return Undefined(), nil
	case TokLBrace:
		if p.inPattern {
			return p.parseObjectPattern()
		}
		return p.parseObjectLiteral()
	case TokLBracket:
		if p.inPattern {
			return p.parseArrayPattern()
		}
		return p.parseArrayLiteral()
	case TokIdentifier:
		name := p.next().Literal
		// --- PATCH: 支持箭头函数变量直接绑定函数值 ---
		if p.peek().Type == TokArrow {
			// 处理 let f = x => ...; 形式
			p.pos-- // 回退，让 parseArrowFunction 重新读取 identifier
			return p.parseArrowFunction()
		}
		val := JSValue{Type: JSIdentifier, String: name}

		// 检查是否是成员访问 obj.prop 或 obj[prop]
		for p.peek().Type == TokDot || p.peek().Type == TokLBracket {
			if p.peek().Type == TokDot {
				p.next() // consume dot
				if p.peek().Type != TokIdentifier {
					return Undefined(), p.errorf("expected property name after dot")
				}
				propName := p.next().Literal
				val = JSValue{Type: JSMember, String: propName, Object: val}
			} else if p.peek().Type == TokLBracket {
				p.next() // consume [
				expr, err := p.ParseExpression()
				if err != nil {
					return expr, err
				}
				_, err = p.expect(TokRBracket)
				if err != nil {
					return Undefined(), err
				}
				// 支持 obj[expr]
				if expr.Type == JSString {
					val = JSValue{Type: JSMember, String: expr.String, Object: val}
				} else if expr.Type == JSNumber {
					val = JSValue{Type: JSMember, Number: expr.Number, Object: val}
				} else {
					return Undefined(), p.errorf("unsupported key type for member access")
				}
			}
		}

		// 只有在不是赋值、调用、成员访问时才解析标识符/成员的值
		if p.peek().Type != TokLParen && p.peek().Type != TokDot && p.peek().Type != TokLBracket &&
			p.peek().Type != TokAssign {
			if val.Type == JSIdentifier {
				if actualVal, ok := p.ctx.global.Get(val.String); ok {
					val = actualVal
				}
			} else if val.Type == JSMember {
				// Resolve member access for non-call, non-assignment contexts
				var resolve func(JSValue) JSValue
				resolve = func(m JSValue) JSValue {
					switch m.Type {
					case JSMember:
						// resolve parent value
						parentRaw := m.Object.(JSValue)
						var parent JSValue
						if parentRaw.Type == JSIdentifier {
							actual, ok := p.ctx.global.Get(parentRaw.String)
							if !ok {
								return Undefined()
							}
							parent = actual
						} else {
							parent = parentRaw
						}
						if parent.Type != JSObject {
							return Undefined()
						}
						objMap := parent.Object.(map[string]JSValue)
						var key string
						if m.String != "" {
							key = m.String
						} else {
							key = strconv.Itoa(int(m.Number))
						}
						v, ok := objMap[key]
						if !ok {
							return Undefined()
						}
						return v
					default:
						return m
					}
				}
				val = resolve(val)
			}
		}
		// 检查是否是函数调用
		if p.peek().Type == TokLParen {
			if val.Type == JSMember {
				obj := val
				propName := obj.String
				// 递归地获取基础对象
				var baseObj JSValue = obj.Object.(JSValue)
				for baseObj.Type == JSMember {
					baseObj = baseObj.Object.(JSValue)
				}

				// 解析基础对象的值
				var thisVal JSValue
				if baseObj.Type == JSIdentifier {
					objVal, ok := p.ctx.global.Get(baseObj.String)
					if !ok {
						return Undefined(), p.errorf("object not found: %s", baseObj.String)
					}
					thisVal = objVal
				} else {
					thisVal = baseObj
				}
				return p.parseCall(propName, thisVal)
			}
			// 普通函数调用，this is Undefined
			return p.parseCall(val.String, Undefined())
		}
		// 不在赋值、调用、成员访问时才解析成员值（已处理在上面）
		return val, nil
	case TokFunction:
		return p.parseFunctionDecl(true)
	case TokNew:
		return p.parseNew()
	case TokSuper:
		p.next()
		return JSValue{Type: JSSuper}, nil
	case TokLParen:
		if p.isArrowFunctionAhead() {
			return p.parseArrowFunction()
		}
		p.next()
		val, err := p.ParseExpression()
		if err != nil {
			return val, err
		}
		_, err = p.expect(TokRParen)
		return val, err
	default:
		return Undefined(), p.errorf("unexpected token: %s", tok.Literal)
	}
}

// parseObjectLiteral parses { key: value, ... } and { key() { ... }, ... }
func (p *Parser) parseObjectLiteral() (JSValue, error) {
	obj := make(map[string]JSValue)
	_, err := p.expect(TokLBrace)
	if err != nil {
		return Undefined(), err
	}
	for p.peek().Type != TokRBrace && p.peek().Type != TokEOF {
		if p.peek().Type != TokIdentifier && p.peek().Type != TokString {
			return Undefined(), p.errorf("expected property name in object literal")
		}
		keyTok := p.next()
		key := keyTok.Literal
		if p.peek().Type == TokLParen {
			// 对象方法简写：key() { ... }
			fn, err := p.parseFunctionDecl(false)
			if err != nil {
				return fn, err
			}
			obj[key] = fn
		} else {
			if p.peek().Type != TokColon {
				return Undefined(), p.errorf("expected ':' after property name in object literal")
			}
			p.next() // consume ':'
			val, err := p.ParseExpression()
			if err != nil {
				return val, err
			}
			obj[key] = val
		}
		if p.peek().Type == TokComma {
			p.next()
		} else {
			break
		}
	}
	_, err = p.expect(TokRBrace)
	if err != nil {
		return Undefined(), err
	}
	return ObjectVal(obj), nil
}

// parseArrayLiteral parses [elem1, elem2, ...]
func (p *Parser) parseArrayLiteral() (JSValue, error) {
	arr := []JSValue{}
	_, err := p.expect(TokLBracket)
	if err != nil {
		return Undefined(), err
	}
	for p.peek().Type != TokRBracket && p.peek().Type != TokEOF {
		val, err := p.ParseExpression()
		if err != nil {
			return val, err
		}
		arr = append(arr, val)
		if p.peek().Type == TokComma {
			p.next()
		} else {
			break
		}
	}
	_, err = p.expect(TokRBracket)
	if err != nil {
		return Undefined(), err
	}
	// For now, represent arrays as objects with numeric keys and length
	obj := make(map[string]JSValue)
	for i, v := range arr {
		obj[strconv.Itoa(i)] = v
	}
	obj["length"] = NumberVal(float64(len(arr)))
	return ObjectVal(obj), nil
}

// parseObjectPattern parses {a, b, c} for destructuring assignment
func (p *Parser) parseObjectPattern() (JSValue, error) {
	obj := make(map[string]JSValue)
	_, err := p.expect(TokLBrace)
	if err != nil {
		return Undefined(), err
	}
	for p.peek().Type != TokRBrace && p.peek().Type != TokEOF {
		if p.peek().Type != TokIdentifier {
			return Undefined(), p.errorf("expected identifier in object pattern")
		}
		keyTok := p.next()
		key := keyTok.Literal
		obj[key] = JSValue{Type: JSIdentifier, String: key}
		if p.peek().Type == TokComma {
			p.next()
		} else {
			break
		}
	}
	_, err = p.expect(TokRBrace)
	if err != nil {
		return Undefined(), err
	}
	return ObjectVal(obj), nil
}

// parseArrayPattern parses [a, b, c] for destructuring assignment
func (p *Parser) parseArrayPattern() (JSValue, error) {
	arr := []string{}
	_, err := p.expect(TokLBracket)
	if err != nil {
		return Undefined(), err
	}
	for p.peek().Type != TokRBracket && p.peek().Type != TokEOF {
		if p.peek().Type != TokIdentifier {
			return Undefined(), p.errorf("expected identifier in array pattern")
		}
		name := p.next().Literal
		arr = append(arr, name)
		if p.peek().Type == TokComma {
			p.next()
		} else {
			break
		}
	}
	_, err = p.expect(TokRBracket)
	if err != nil {
		return Undefined(), err
	}
	// Represent as object with numeric keys (string) and identifier values
	obj := make(map[string]JSValue)
	for i, name := range arr {
		obj[strconv.Itoa(i)] = JSValue{Type: JSIdentifier, String: name}
	}
	obj["length"] = NumberVal(float64(len(arr)))
	return ObjectVal(obj), nil
}
