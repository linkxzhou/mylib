package simplejs

// parseFunctionDecl parses function declaration/expr: function (a, b) { ... }
func (p *Parser) parseFunctionDecl(requireFunction bool) (JSValue, error) {
	if requireFunction {
		p.next() // consume 'function'
	}
	var name string
	if p.peek().Type == TokIdentifier {
		name = p.next().Literal
	}
	_, err := p.expect(TokLParen)
	if err != nil {
		return Undefined(), err
	}
	var params []string
	for p.peek().Type != TokRParen {
		if p.peek().Type != TokIdentifier {
			return Undefined(), p.errorf("expected parameter name")
		}
		params = append(params, p.next().Literal)
		if p.peek().Type == TokComma {
			p.next()
		} else {
			break
		}
	}
	_, err = p.expect(TokRParen)
	if err != nil {
		return Undefined(), err
	}
	_, err = p.expect(TokLBrace)
	if err != nil {
		return Undefined(), err
	}
	bodyStart := p.pos - 1 // include the opening brace
	braceCount := 1
	for i := p.pos; i < len(p.tokens); i++ {
		if p.tokens[i].Type == TokLBrace {
			braceCount++
		} else if p.tokens[i].Type == TokRBrace {
			braceCount--
			if braceCount == 0 {
				bodyEnd := i
				funcTokens := make([]Token, bodyEnd-bodyStart+1)
				copy(funcTokens, p.tokens[bodyStart:bodyEnd+1])
				info := &FunctionInfo{
					ParamNames: params,
					Body:       funcTokens,
					Closure:    p.ctx.global,
				}
				fnVal := JSValue{Type: JSFunction, FuncInfo: info}
				if name != "" {
					p.ctx.global.Set(name, fnVal)
				}
				p.pos = bodyEnd + 1
				p.debug("FunctionDecl: name=%v, params=%v", name, params)
				return fnVal, nil
			}
		}
	}
	return Undefined(), p.errorf("unterminated function body")
}

// isArrowFunctionAhead 检查是否为箭头函数 (a, b) =>
func (p *Parser) isArrowFunctionAhead() bool {
	if p.peek().Type != TokLParen {
		return false
	}
	// 保存当前位置
	savedPos := p.pos
	defer func() { p.pos = savedPos }()

	p.next() // consume '('
	// 检查参数列表
	for p.peek().Type != TokRParen && p.peek().Type != TokEOF {
		if p.peek().Type != TokIdentifier {
			return false
		}
		p.next() // consume parameter name
		if p.peek().Type == TokComma {
			p.next() // consume comma
		} else {
			break
		}
	}
	if p.peek().Type != TokRParen {
		return false
	}
	p.next() // consume ')'
	// 检查是否有箭头
	return p.peek().Type == TokArrow
}

// parseArrowFunction parses arrow functions: (a, b) => expr or a => expr
func (p *Parser) parseArrowFunction() (JSValue, error) {
	var params []string
	// 检查是否是带括号的参数列表
	if p.peek().Type == TokLParen {
		p.next() // consume '('
		for p.peek().Type != TokRParen {
			if p.peek().Type != TokIdentifier {
				return Undefined(), p.errorf("expected parameter name")
			}
			params = append(params, p.next().Literal)
			if p.peek().Type == TokComma {
				p.next()
			} else {
				break
			}
		}
		_, err := p.expect(TokRParen)
		if err != nil {
			return Undefined(), err
		}
	} else if p.peek().Type == TokIdentifier {
		// 单个参数不带括号
		params = append(params, p.next().Literal)
	} else {
		return Undefined(), p.errorf("expected parameter(s) for arrow function")
	}

	// 箭头
	_, err := p.expect(TokArrow)
	if err != nil {
		return Undefined(), err
	}

	// 函数体
	var bodyTokens []Token
	if p.peek().Type == TokLBrace {
		// 块级函数体 { ... }
		p.next() // consume '{'
		bodyStart := p.pos - 1
		braceCount := 1
		for i := p.pos; i < len(p.tokens) && braceCount > 0; i++ {
			if p.tokens[i].Type == TokLBrace {
				braceCount++
			} else if p.tokens[i].Type == TokRBrace {
				braceCount--
				if braceCount == 0 {
					bodyEnd := i
					bodyTokens = make([]Token, bodyEnd-bodyStart+1)
					copy(bodyTokens, p.tokens[bodyStart:bodyEnd+1])
					p.pos = bodyEnd + 1
					break
				}
			}
		}
	} else {
		// 表达式函数体 => expr
		// 捕获完整表达式，不仅仅是单个token
		exprStart := p.pos
		exprLevel := 0
		for i := p.pos; i < len(p.tokens); i++ {
			tok := p.tokens[i]
			// 处理括号层级，保证捕获完整的表达式
			if tok.Type == TokLParen || tok.Type == TokLBrace || tok.Type == TokLBracket {
				exprLevel++
			} else if tok.Type == TokRParen || tok.Type == TokRBrace || tok.Type == TokRBracket {
				exprLevel--
			} else if exprLevel == 0 && (tok.Type == TokSemicolon || tok.Type == TokComma ||
				tok.Type == TokRParen || tok.Type == TokColon) {
				// 在表达式边界终止
				bodyEnd := i - 1
				if bodyEnd >= exprStart {
					bodyTokens = make([]Token, bodyEnd-exprStart+1)
					copy(bodyTokens, p.tokens[exprStart:bodyEnd+1])
					p.pos = i
					if tok.Type != TokSemicolon && tok.Type != TokComma {
						p.pos = i // 不消费语句终止符
					}
					break
				}
			}

			// 如果到了最后一个token，也要终止
			if i == len(p.tokens)-1 {
				bodyTokens = make([]Token, i-exprStart+1)
				copy(bodyTokens, p.tokens[exprStart:i+1])
				p.pos = i + 1
				break
			}
		}

		// 如果没有正确捕获表达式，至少捕获一个token
		if len(bodyTokens) == 0 {
			bodyTokens = []Token{p.tokens[p.pos]}
			p.pos++
		}
	}

	// 保存当前作用域，让箭头函数能访问外部变量
	currentScope := p.ctx.global

	info := &FunctionInfo{
		ParamNames: params,
		Body:       bodyTokens,
		Closure:    currentScope, // 捕获当前作用域作为闭包
		IsArrow:    true,
		This:       JSValue{Type: JSUndefined}, // 箭头函数会继承调用时的this
	}
	p.debug("ArrowFunction: params=%v", params)
	return JSValue{Type: JSFunction, FuncInfo: info}, nil
}

// parseClass parses class Foo [extends Bar] { ... }
func (p *Parser) parseClass() (JSValue, error) {
	p.next() // consume 'class'
	if p.peek().Type != TokIdentifier {
		return Undefined(), p.errorf("expected class name")
	}
	className := p.next().Literal
	var superProto *JSValue = nil
	if p.peek().Type == TokExtends {
		p.next()
		if p.peek().Type != TokIdentifier {
			return Undefined(), p.errorf("expected superclass name after extends")
		}
		superName := p.next().Literal
		sup, ok := p.ctx.global.Get(superName)
		if !ok || sup.Type != JSFunction || sup.FuncInfo == nil {
			return Undefined(), p.errorf("superclass %s not found or not a function", superName)
		}
		if sup.Proto != nil {
			superProto = sup.Proto
		}
	}

	_, err := p.expect(TokLBrace)
	if err != nil {
		return Undefined(), err
	}

	// 解析类方法
	methods := make(map[string]JSValue)
	for p.peek().Type != TokRBrace && p.peek().Type != TokEOF {
		if p.peek().Type != TokIdentifier {
			return Undefined(), p.errorf("expected method name")
		}
		methodName := p.next().Literal
		if methodName == "constructor" && p.peek().Type == TokLParen {
			// 构造函数
			fn, err := p.parseFunctionDecl(false)
			if err != nil {
				return fn, err
			}
			methods["constructor"] = fn
		} else if p.peek().Type == TokLParen {
			// 普通方法
			fn, err := p.parseFunctionDecl(false)
			if err != nil {
				return fn, err
			}
			methods[methodName] = fn
			p.debug("Class method: %v", methodName)
		} else {
			return Undefined(), p.errorf("expected method definition")
		}
	}
	_, err = p.expect(TokRBrace)
	if err != nil {
		return Undefined(), err
	}
	// 构造器
	constructor, ok := methods["constructor"]
	if !ok {
		constructor = JSValue{Type: JSFunction, FuncInfo: &FunctionInfo{ParamNames: []string{}, Body: []Token{}, Closure: p.ctx.global}}
	}
	// 类本身是一个工厂函数
	classFn := JSValue{Type: JSFunction, FuncInfo: constructor.FuncInfo}
	protoObj := JSValue{Type: JSObject, Object: methods, Proto: superProto}
	classFn.Proto = &protoObj
	p.ctx.global.Set(className, classFn)
	p.debug("ClassDecl: name=%v, methods=%v", className, methods)
	return classFn, nil
}

// parseNew parses new expressions: new Constructor(args)
func (p *Parser) parseNew() (JSValue, error) {
	p.next() // consume 'new'

	// Get constructor function
	if p.peek().Type != TokIdentifier {
		return Undefined(), p.errorf("expected constructor name after new")
	}

	constructorName := p.next().Literal
	constructor, ok := p.ctx.global.Get(constructorName)
	if !ok || constructor.Type != JSFunction {
		return Undefined(), p.errorf("constructor %s not found or not a function", constructorName)
	}

	// Parse arguments
	_, err := p.expect(TokLParen)
	if err != nil {
		return Undefined(), err
	}

	var args []JSValue
	for p.peek().Type != TokRParen && p.peek().Type != TokEOF {
		arg, err := p.ParseExpression()
		if err != nil {
			return Undefined(), err
		}
		args = append(args, arg)
		if p.peek().Type == TokComma {
			p.next() // consume comma
		} else {
			break
		}
	}

	_, err = p.expect(TokRParen)
	if err != nil {
		return Undefined(), err
	}

	// Create a new object with the constructor's prototype
	instance := JSValue{Type: JSObject, Object: make(map[string]JSValue)}
	if constructor.Proto != nil {
		instance.Proto = constructor.Proto
	}
	// --- PATCH: set instance.__proto__ to prototype for JS semantics ---
	if constructor.Proto != nil {
		protoObj := *constructor.Proto
		if protoObj.Type == JSObject {
			instance.Object.(map[string]JSValue)["__proto__"] = protoObj
		}
	}

	// Set 'this' to the new instance and call the constructor
	if constructor.FuncInfo != nil {
		// Create a new scope for the constructor call
		oldScope := p.ctx.global
		funcScope := NewScope(constructor.FuncInfo.Closure)
		p.ctx.global = funcScope

		// Bind 'this' to the new instance
		p.ctx.global.Set("this", instance)
		// PATCH: also bind 'this' as reference to instance.Object so property assignment works
		p.ctx.global.vars["this"] = instance

		// Bind arguments to parameter names
		for i, param := range constructor.FuncInfo.ParamNames {
			if i < len(args) {
				p.ctx.global.Set(param, args[i])
			} else {
				p.ctx.global.Set(param, Undefined())
			}
		}

		// Execute constructor body
		parser := NewParser(constructor.FuncInfo.Body, p.ctx)
		_, err := parser.ParseProgram()

		// Restore original scope
		p.ctx.global = oldScope
		// PATCH: update instance from scope after constructor call
		if thisVal, ok := funcScope.Get("this"); ok && thisVal.Type == JSObject {
			instance = thisVal
		}

		if err != nil {
			return Undefined(), err
		}
	}
	p.debug("New: constructor=%v, args=%v", constructorName, args)
	return instance, nil
}

// parseCall handles function call expressions: fn(...)
func (p *Parser) parseCall(name string, this JSValue) (JSValue, error) {
	_, err := p.expect(TokLParen)
	if err != nil {
		return Undefined(), err
	}

	// 收集参数
	var args []JSValue
	for p.peek().Type != TokRParen && p.peek().Type != TokEOF {
		arg, err := p.ParseExpression()
		if err != nil {
			return Undefined(), err
		}
		args = append(args, arg)
		if p.peek().Type == TokComma {
			p.next() // consume comma
			continue
		}
		break
	}

	_, err = p.expect(TokRParen)
	if err != nil {
		return Undefined(), err
	}

	// 获取函数值
	var fnVal JSValue
	var ok bool
	// 如果 this 不是 Undefined，说明是对象方法调用
	if this.Type != JSUndefined && this.Type == JSObject {
		// 从对象中获取方法
		objMap := this.Object.(map[string]JSValue)
		fnVal, ok = objMap[name]

		// 如果对象中没有该方法，尝试从原型链中查找
		if !ok && this.Proto != nil {
			protoObj := *this.Proto
			if protoObj.Type == JSObject {
				protoMap := protoObj.Object.(map[string]JSValue)
				fnVal, ok = protoMap[name]
			}
		}
	} else {
		// 普通函数调用
		fnVal, ok = p.ctx.global.Get(name)
	}

	if !ok || fnVal.Type != JSFunction {
		return Undefined(), p.errorf("not a function: %s", name)
	}

	// 兼容 Go 注册的原生函数（FuncInfo 可能为 nil）
	if fnVal.FuncInfo == nil && fnVal.Function != nil {
		// 直接调用 Go 函数
		result := fnVal.Function(args...)
		return result, nil
	}

	// 创建新的作用域
	oldScope := p.ctx.global
	funcScope := NewScope(fnVal.FuncInfo.Closure) // Restore closure scope for function calls
	p.ctx.global = funcScope
	defer func() { p.ctx.global = oldScope }()

	// 绑定 this
	if fnVal.FuncInfo.IsArrow {
		// 箭头函数继承外部的 this
		if fnVal.FuncInfo.This.Type != JSUndefined {
			p.ctx.global.Set("this", fnVal.FuncInfo.This)
		}
	} else {
		// 普通函数的 this 绑定到调用者
		p.ctx.global.Set("this", this)
	}

	// 绑定参数
	for i, param := range fnVal.FuncInfo.ParamNames {
		if i < len(args) {
			p.ctx.global.Set(param, args[i])
		} else {
			p.ctx.global.Set(param, Undefined())
		}
	}

	// 执行函数体
	parser := NewParser(fnVal.FuncInfo.Body, p.ctx)
	var result JSValue = Undefined()
	var returnPanic interface{} = nil

	// 处理可能的 return 语句
	func() {
		defer func() {
			if r := recover(); r != nil {
				if ret, ok := r.(ReturnPanic); ok {
					result = ret.Value
					returnPanic = nil
				} else {
					returnPanic = r
				}
			}
		}()
		result, err = parser.ParseProgram()
	}()
	if returnPanic != nil {
		panic(returnPanic)
	}
	if err != nil {
		return Undefined(), err
	}
	p.debug("Call: name=%v, args=%v, result=%v", name, args, result.ToString())
	return result, nil
}

// BreakPanic 用于 break 语句的 panic
type BreakPanic struct{}
