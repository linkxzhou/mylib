package simplejs

import (
	"strconv"
)

// parseAssignment handles identifier = expr (recursive, right-associative)
func (p *Parser) parseAssignment() (JSValue, error) {
	left, err := p.parseTernary()
	if err != nil {
		return left, err
	}
	// Only allow simple assignment, not +=, -=, etc.
	if p.peek().Type == TokAssign {
		p.next()                          // consume '='
		right, err := p.parseAssignment() // right-associative
		if err != nil {
			return right, err
		}
		if left.Type == JSIdentifier {
			p.ctx.global.SetInChain(left.String, right)
			p.debug("Simple assignment: %v = %v", left.ToString(), right.ToString())
			return right, nil
		} else if left.Type == JSMember {
			obj := left
			for obj.Type == JSMember {
				obj = obj.Object.(JSValue) // type assertion: for JSMember, Object is a JSValue
			}
			if obj.Type == JSIdentifier {
				objVal, ok := p.ctx.global.Get(obj.String)
				if !ok {
					return Undefined(), p.errorf("object not found: %s", obj.String)
				}
				obj = objVal
			}
			if obj.Type != JSObject {
				return Undefined(), p.errorf("invalid assignment target (not object)")
			}
			objMap := obj.Object.(map[string]JSValue)
			var key string
			if left.String != "" {
				key = left.String
			} else {
				key = strconv.Itoa(int(left.Number))
			}
			objMap[key] = right
			// --- PATCH: update array length if needed ---
			if obj.IsArray {
				idx := 0
				if left.String != "" {
					if i, err := strconv.Atoi(left.String); err == nil {
						idx = i
					}
				} else {
					idx = int(left.Number)
				}
				if l, ok := objMap["length"]; ok {
					if idx+1 > int(l.ToNumber()) {
						objMap["length"] = NumberVal(float64(idx + 1))
					}
				} else {
					objMap["length"] = NumberVal(float64(idx + 1))
				}
			}
			p.debug("Object index assignment: obj[%v] = %v", key, right.ToString())
			return right, nil
		}
		return Undefined(), p.errorf("invalid assignment target")
	}
	return left, nil
}

// parseLogicalOr handles ||
func (p *Parser) parseLogicalOr() (JSValue, error) {
	left, err := p.parseLogicalAnd()
	if err != nil {
		return left, err
	}
	for p.peek().Type == TokLogicalOr {
		p.next()
		right, err := p.parseLogicalAnd()
		if err != nil {
			return left, err
		}
		p.debug("Logical OR: %v || %v", left.ToString(), right.ToString())
		left = BoolVal(left.ToBool() || right.ToBool())
		p.debug("Result of OR: %v", left.ToString())
	}
	return left, nil
}

// parseLogicalAnd handles &&
func (p *Parser) parseLogicalAnd() (JSValue, error) {
	left, err := p.parseEquality()
	if err != nil {
		return left, err
	}
	for p.peek().Type == TokLogicalAnd {
		p.next()
		right, err := p.parseEquality()
		if err != nil {
			return left, err
		}
		p.debug("Logical AND: %v && %v", left.ToString(), right.ToString())
		left = BoolVal(left.ToBool() && right.ToBool())
		p.debug("Result of AND: %v", left.ToString())
	}
	return left, nil
}

// parseEquality handles ==, !=, ===, !==
func (p *Parser) parseEquality() (JSValue, error) {
	left, err := p.parseRelational()
	if err != nil {
		return left, err
	}
	for p.peek().Type == TokEqual || p.peek().Type == TokNotEqual || p.peek().Type == TokStrictEqual || p.peek().Type == TokNotStrictEqual {
		op := p.next().Type
		right, err := p.parseRelational()
		if err != nil {
			return right, err
		}
		p.debug("Equality: %v %v %v", left.ToString(), op, right.ToString())
		var result bool
		switch op {
		case TokEqual:
			result = left.ToString() == right.ToString()
		case TokNotEqual:
			result = left.ToString() != right.ToString()
		case TokStrictEqual:
			result = (left.Type == right.Type) && (left.ToString() == right.ToString())
		case TokNotStrictEqual:
			result = (left.Type != right.Type) || (left.ToString() != right.ToString())
		}
		p.debug("Result of equality: %v", result)
		left = BoolVal(result)
	}
	return left, nil
}

// parseRelational handles <, >, <=, >=
func (p *Parser) parseRelational() (JSValue, error) {
	left, err := p.parseAddSub()
	if err != nil {
		return left, err
	}
	for p.peek().Type == TokLT || p.peek().Type == TokGT ||
		p.peek().Type == TokLTE || p.peek().Type == TokGTE {
		op := p.next().Type
		right, err := p.parseAddSub()
		if err != nil {
			return right, err
		}
		p.debug("Relational: %v %v %v", left.ToString(), op, right.ToString())
		var result bool
		switch op {
		case TokLT:
			result = left.ToNumber() < right.ToNumber()
		case TokGT:
			result = left.ToNumber() > right.ToNumber()
		case TokLTE:
			result = left.ToNumber() <= right.ToNumber()
		case TokGTE:
			result = left.ToNumber() >= right.ToNumber()
		}
		p.debug("Result of relational: %v", result)
		left = BoolVal(result)
	}
	return left, nil
}

func resolveIfIdentifier(ctx *RunContext, v JSValue) JSValue {
	if v.Type == JSIdentifier {
		if actual, ok := ctx.global.Get(v.String); ok {
			return actual
		}
	}
	return v
}

// parseUnary handles unary operators like !, -, delete
func (p *Parser) parseUnary() (JSValue, error) {
	tok := p.peek()
	switch tok.Type {
	case TokLogicalNot:
		p.next()
		operand, err := p.parseUnary()
		if err != nil {
			return operand, err
		}
		return BoolVal(!operand.ToBool()), nil
	case TokMinus:
		p.next()
		operand, err := p.parseUnary()
		if err != nil {
			return operand, err
		}
		return NumberVal(-operand.ToNumber()), nil
	case TokDelete:
		p.next() // consume delete
		operand, err := p.parseUnary()
		if err != nil {
			return operand, err
		}

		// delete 只能用于对象属性
		if operand.Type == JSMember {
			// 获取对象
			obj := operand.Object.(JSValue)
			// 解析对象，直到找到基础对象
			for obj.Type == JSMember {
				obj = obj.Object.(JSValue)
			}

			// 解析基础对象
			var baseObj JSValue
			if obj.Type == JSIdentifier {
				val, ok := p.ctx.global.Get(obj.String)
				if !ok {
					return BoolVal(false), nil // 对象不存在，返回true
				}
				baseObj = val
			} else {
				baseObj = obj
			}

			if baseObj.Type != JSObject {
				return BoolVal(false), nil // 不是对象，无法删除
			}

			// 删除属性
			objMap := baseObj.Object.(map[string]JSValue)
			var key string
			if operand.String != "" {
				key = operand.String
			} else {
				key = strconv.Itoa(int(operand.Number))
			}

			_, exists := objMap[key]
			if exists {
				delete(objMap, key)
				return BoolVal(true), nil
			}
			return BoolVal(false), nil
		}

		return BoolVal(false), nil // 不是对象属性，无法删除
	default:
		return p.parsePrimary()
	}
}

// parsePostfix handles postfix ++ and --
func (p *Parser) parsePostfix() (JSValue, error) {
	left, err := p.parseUnary()
	if err != nil {
		return left, err
	}
	// REMOVED: ++ and -- support
	return left, nil
}

// parseAddSub handles + and -.
func (p *Parser) parseAddSub() (JSValue, error) {
	left, err := p.parseMulDiv()
	if err != nil {
		return left, err
	}
	for p.peek().Type == TokPlus || p.peek().Type == TokMinus {
		op := p.next().Type
		right, err := p.parseMulDiv()
		if err != nil {
			return right, err
		}
		p.debug("AddSub: %v %v %v", left.ToString(), op, right.ToString())
		leftVal := resolveIfIdentifier(p.ctx, left)
		rightVal := resolveIfIdentifier(p.ctx, right)
		switch op {
		case TokPlus:
			if leftVal.Type == JSString || rightVal.Type == JSString {
				left = StringVal(leftVal.ToString() + rightVal.ToString())
			} else {
				left = NumberVal(leftVal.ToNumber() + rightVal.ToNumber())
			}
		case TokMinus:
			left = NumberVal(leftVal.ToNumber() - rightVal.ToNumber())
		}
		p.debug("Result of AddSub: %v", left.ToString())
	}
	return left, nil
}

// parseMulDiv handles * and /.
func (p *Parser) parseMulDiv() (JSValue, error) {
	left, err := p.parsePostfix()
	if err != nil {
		return left, err
	}
	for p.peek().Type == TokAsterisk || p.peek().Type == TokSlash {
		op := p.next().Type
		right, err := p.parsePostfix()
		if err != nil {
			return right, err
		}
		p.debug("MulDiv: %v %v %v", left.ToString(), op, right.ToString())
		switch op {
		case TokAsterisk:
			left = NumberVal(left.ToNumber() * right.ToNumber())
		case TokSlash:
			left = NumberVal(left.ToNumber() / right.ToNumber())
		}
		p.debug("Result of MulDiv: %v", left.ToString())
	}
	return left, nil
}

// parseTernary handles ternary operator (cond ? expr1 : expr2)
func (p *Parser) parseTernary() (JSValue, error) {
	left, err := p.parseLogicalOr()
	if err != nil {
		return left, err
	}
	if p.peek().Type == TokQuestion {
		p.next() // consume ?
		trueExpr, err := p.parseAssignment()
		if err != nil {
			return trueExpr, err
		}
		if p.peek().Type != TokColon {
			return Undefined(), p.errorf("expected : in ternary expression")
		}
		p.next() // consume :
		falseExpr, err := p.parseAssignment()
		if err != nil {
			return falseExpr, err
		}
		if left.ToBool() {
			return trueExpr, nil
		}
		return falseExpr, nil
	}
	return left, nil
}
