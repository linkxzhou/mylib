package simplejs

// 该文件实现函数与类的求值逻辑

func evalClassDecl(cd *ClassDecl, ctx *RunContext) (JSValue, error) {
	// 解析继承
	var superCtor JSValue = Undefined()
	if cd.SuperClass != nil {
		sc, err := evalExpression(cd.SuperClass, ctx)
		if err != nil {
			return Undefined(), err
		}
		superCtor = sc
	}
	
	// 构建原型对象（proto）
	proto := make(map[string]JSValue)
	
	// 如果有父类，继承父类的原型方法
	if superCtor.Type == JSFunction && superCtor.Proto != nil {
		if superProto := superCtor.Proto.ToObject(); superProto != nil {
			for k, v := range superProto {
				proto[k] = v
			}
		}
	}
	
	// 添加类自己的方法到原型对象
	for _, method := range cd.Body {
		fn := method.Value
		if fn != nil && fn.Name != nil && fn.Name.Name != "constructor" {
			// 创建方法函数，确保正确传递this和super
			methodName := fn.Name.Name
			fnVal := FunctionVal(func(args ...JSValue) JSValue {
				// 第一个参数是this对象
				if len(args) > 0 && args[0].Type == JSObject {
					return evalFunctionDecl(fn, args[0], superCtor, ctx, args[1:]...)
				}
				return evalFunctionDecl(fn, Undefined(), superCtor, ctx, args...)
			})
			proto[methodName] = fnVal
		}
	}
	
	// 创建原型对象
	protoObj := JSValue{Type: JSObject, Object: proto}
	
	// 创建构造函数
	ctor := FunctionVal(func(args ...JSValue) JSValue {
		// 创建新对象实例
		obj := JSValue{
			Type: JSObject, 
			Object: make(map[string]JSValue),
		}
		
		// 设置原型链
		obj.Proto = &protoObj
		
		// 保存旧的作用域
		oldScope := ctx.global
		ctx.global = NewScope(oldScope)
		defer func() { ctx.global = oldScope }()
		
		// 设置this和super
		ctx.global.Set("this", obj)
		if superCtor.Type == JSFunction {
			ctx.global.Set("super", superCtor)
		}
		
		// 调用constructor方法（如果有）
		var constructorFound bool
		for _, method := range cd.Body {
			fn := method.Value
			if fn != nil && fn.Name != nil && fn.Name.Name == "constructor" {
				_ = evalFunctionDecl(fn, obj, superCtor, ctx, args...)
				constructorFound = true
				break
			}
		}
		
		// 如果没有找到构造函数但有父类，调用父类构造函数
		if !constructorFound && superCtor.Type == JSFunction {
			superCtor.Function(args...)
		}
		
		return obj
	})
	
	// 设置构造函数的原型
	ctor.Proto = &protoObj
	
	// 如果有类名，将构造函数绑定到全局作用域
	if cd.Name != nil {
		ctx.global.Set(cd.Name.Name, ctor)
	}
	
	return ctor, nil
}

func evalFunctionDecl(fd *FunctionDecl, thisVal JSValue, superCtor JSValue, ctx *RunContext, args ...JSValue) JSValue {
	oldScope := ctx.global
	defer func() { ctx.global = oldScope }()
	ctx.global = NewScope(oldScope)
	if thisVal.Type != JSUndefined {
		ctx.global.Set("this", thisVal)
	}
	if superCtor.Type == JSFunction {
		ctx.global.Set("super", superCtor)
	}
	for i, param := range fd.Params {
		if i < len(args) {
			ctx.global.Set(param.Name, args[i])
		} else {
			ctx.global.Set(param.Name, Undefined())
		}
	}
	// Execute function body and catch return panic
	result := Undefined()
	func() {
		defer func() {
			if r := recover(); r != nil {
				if rp, ok := r.(ReturnPanic); ok {
					result = rp.Value
					return
				}
				panic(r)
			}
		}()
		val, err := evalStatement(fd.Body, ctx)
		if err == nil {
			result = val
		}
	}()
	return result
}
