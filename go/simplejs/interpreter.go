package simplejs

import (
	"fmt"
	"errors"
)

// Interpreter 用于执行 AST 节点
// 设计：每种 Statement/Expression 类型都实现 Accept(visitor Interpreter) 或 Eval(ctx *RunContext) 方法
// 这里采用简单的 switch+类型断言实现

func (prog *Program) Eval(ctx *RunContext) (JSValue, error) {
	var result JSValue = Undefined()
	// 捕获函数中的 return，使顶层 Eval 不 panic
	defer func() {
		if r := recover(); r != nil {
			if rp, ok := r.(ReturnPanic); ok {
				result = rp.Value
				return
			}
			panic(r)
		}
	}()
	for _, stmt := range prog.Body {
		val, err := evalStatement(stmt, ctx)
		if err != nil {
			return val, err
		}
		result = val
	}
	return result, nil
}

func evalStatement(stmt Statement, ctx *RunContext) (JSValue, error) {
	switch s := stmt.(type) {
	case *VarDecl:
		var val JSValue = Undefined()
		if s.Init != nil {
			v, err := evalExpression(s.Init, ctx)
			if err != nil {
				return val, err
			}
			val = v
		}
		// Handle destructuring
		switch pattern := s.Name.(type) {
		case *Identifier:
			ctx.global.Set(pattern.Name, val)
		case *ObjectLiteral:
			if err := destructureObjectPattern(pattern, val, ctx); err != nil {
				return Undefined(), err
			}
		case *ArrayLiteral:
			if err := destructureArrayPattern(pattern, val, ctx); err != nil {
				return Undefined(), err
			}
		default:
			return Undefined(), fmt.Errorf("unsupported destructuring pattern: %T", s.Name)
		}
		return val, nil
	case *BlockStmt:
		oldScope := ctx.global
		ctx.global = NewScope(oldScope)
		var result JSValue = Undefined()
		for _, stmt := range s.Body {
			val, err := evalStatement(stmt, ctx)
			if err != nil {
				ctx.global = oldScope
				return val, err
			}
			result = val
		}
		ctx.global = oldScope
		return result, nil
	case *IfStmt:
		cond, err := evalExpression(s.Test, ctx)
		if err != nil {
			return Undefined(), err
		}
		if cond.ToBool() {
			return evalStatement(s.Consequent, ctx)
		} else if s.Alternate != nil {
			return evalStatement(s.Alternate, ctx)
		}
		return Undefined(), nil
	case *WhileStmt:
		// while loop execution
		var result JSValue = Undefined()
		for {
			cond, err := evalExpression(s.Test, ctx)
			if err != nil {
				return Undefined(), err
			}
			if !cond.ToBool() {
				break
			}
			val, err := evalStatement(s.Body, ctx)
			if err == ErrBreak {
				break
			}
			if err != nil {
				return val, err
			}
			result = val
		}
		return result, nil
	case *ForStmt:
		// for loop execution
		// initialization
		if s.Init != nil {
			_, err := evalStatement(s.Init, ctx)
			if err != nil && err != ErrBreak {
				return Undefined(), err
			}
		}
		var result JSValue = Undefined()
		for {
			// test condition
			if s.Test != nil {
				cond, err := evalExpression(s.Test, ctx)
				if err != nil {
					return Undefined(), err
				}
				if !cond.ToBool() {
					break
				}
			}
			// execute body
			val, err := evalStatement(s.Body, ctx)
			if err == ErrBreak {
				break
			}
			if err != nil {
				return val, err
			}
			// update expression
			if s.Update != nil {
				_, err := evalExpression(s.Update, ctx)
				if err != nil {
					return Undefined(), err
				}
			}
			result = val
		}
		return result, nil
	case *ReturnStmt:
		var val JSValue = Undefined()
		if s.Argument != nil {
			v, err := evalExpression(s.Argument, ctx)
			if err != nil {
				return val, err
			}
			val = v
		}
		panic(ReturnPanic{Value: val})
	case *TryCatchStmt:
		return evalTryCatch(s, ctx)
	case *ThrowStmt:
		arg, err := evalExpression(s.Argument, ctx)
		if err != nil {
			return Undefined(), err
		}
		// throw exception by panic to be caught in try-catch
		panic(&JSException{Value: arg})
	case *BreakStmt:
		// break statement
		return Undefined(), ErrBreak
	case *ExpressionStmt:
		// Evaluate expression statements (e.g., assignment, function calls)
		return evalExpression(s.Expr, ctx)
	case *ClassDecl:
		return evalClassDecl(s, ctx)
	case *FunctionDecl:
		// Register function declaration in global scope
		fnVal := FunctionVal(func(args ...JSValue) JSValue {
			// thisVal and superCtor undefined for top-level functions
			return evalFunctionDecl(s, Undefined(), Undefined(), ctx, args...)
		})
		ctx.global.Set(s.Name.Name, fnVal)
		return Undefined(), nil
	default:
		return Undefined(), nil // TODO: 其它语句类型
	}
}

func evalExpression(expr Expression, ctx *RunContext) (JSValue, error) {
	switch e := expr.(type) {
	case *Literal:
		return convertLiteral(e), nil
	case *Identifier:
		val, ok := ctx.global.Get(e.Name)
		if !ok {
			return Undefined(), nil
		}
		return val, nil
	case *BinaryExpr:
		left, err := evalExpression(e.Left, ctx)
		if err != nil {
			return Undefined(), err
		}
		right, err := evalExpression(e.Right, ctx)
		if err != nil {
			return Undefined(), err
		}
		return evalBinaryExpr(e.Op, left, right), nil
	case *UnaryExpr:
		val, err := evalExpression(e.X, ctx)
		if err != nil {
			return Undefined(), err
		}
		return evalUnaryExpr(e.Op, val), nil
	case *MemberExpr:
		// 特殊处理super关键字
		if id, ok := e.Object.(*Identifier); ok && id.Name == "super" {
			// 获取super对象
			superObj, found := ctx.global.Get("super")
			if !found || superObj.Type == JSUndefined {
				return Undefined(), nil
			}

			// 获取属性名
			var key string
			if e.Computed {
				v, err := evalExpression(e.Property, ctx)
				if err != nil {
					return Undefined(), err
				}
				key = v.ToString()
			} else if propId, ok := e.Property.(*Identifier); ok {
				key = propId.Name
			}

			// 从super对象获取方法
			superMap := superObj.ToObject()
			if val, ok := superMap[key]; ok {
				return val, nil
			}
			return Undefined(), nil
		}

		// 正常对象属性访问
		objVal, err := evalExpression(e.Object, ctx)
		if err != nil {
			return Undefined(), err
		}

		var key string
		if e.Computed {
			v, err2 := evalExpression(e.Property, ctx)
			if err2 != nil {
				return Undefined(), err2
			}
			key = v.ToString()
		} else if id, ok := e.Property.(*Identifier); ok {
			key = id.Name
		}

		// 先查实例自身属性
		m := objVal.ToObject()
		if m != nil {
			if val, ok := m[key]; ok {
				return val, nil
			}
		}

		// 递归查找原型链
		cur := objVal.Proto
		for cur != nil {
			protoMap := cur.ToObject()
			if val, ok := protoMap[key]; ok {
				return val, nil
			}
			cur = cur.Proto
		}
		return Undefined(), nil
	case *NewExpr:
		ctorVal, err := evalExpression(e.Callee, ctx)
		if err != nil {
			return Undefined(), err
		}
		fn := ctorVal.ToFunction()
		return fn(evalArguments(e.Arguments, ctx)...), nil
	case *CallExpr:
		// 方法调用 this 绑定 或 普通函数调用
		var thisVal JSValue

		// 检查是否是super.method()调用
		if memberExpr, ok := e.Callee.(*MemberExpr); ok {
			if id, ok2 := memberExpr.Object.(*Identifier); ok2 && id.Name == "super" {
				// 对于super.method()调用，this应该是当前对象
				thisVal, _ = ctx.global.Get("this")

				// 获取super对象
				superObj, found := ctx.global.Get("super")
				if !found || superObj.Type == JSUndefined {
					return Undefined(), nil
				}

				// 获取方法名
				var methodName string
				if memberExpr.Computed {
					v, err := evalExpression(memberExpr.Property, ctx)
					if err != nil {
						return Undefined(), err
					}
					methodName = v.ToString()
				} else if propId, ok := memberExpr.Property.(*Identifier); ok {
					methodName = propId.Name
				}

				// 从super对象获取方法
				superMap := superObj.ToObject()
				if method, ok := superMap[methodName]; ok && method.IsFunction() {
					// 直接调用方法，确保this绑定正确
					args := evalArguments(e.Arguments, ctx)

					// 捕获 return panic
					var res JSValue
					func() {
						defer func() {
							if r := recover(); r != nil {
								if rp, ok := r.(ReturnPanic); ok {
									res = rp.Value
									return
								}
								panic(r)
							}
						}()

						// 调用父类方法，传入子类实例作为this
						res = method.Function(append([]JSValue{thisVal}, args...)...)
					}()
					return res, nil
				}
				return Undefined(), nil
			} else {
				// 普通方法调用
				var err error
				thisVal, err = evalExpression(memberExpr.Object, ctx)
				if err != nil {
					return Undefined(), err
				}
			}
		}

		fnVal, err := evalExpression(e.Callee, ctx)
		if err != nil {
			return Undefined(), err
		}
		fn := fnVal.ToFunction()
		args := evalArguments(e.Arguments, ctx)

		// 捕获 return panic
		var res JSValue
		func() {
			defer func() {
				if r := recover(); r != nil {
					if rp, ok := r.(ReturnPanic); ok {
						res = rp.Value
						return
					}
					panic(r)
				}
			}()

			if thisVal.Type != JSUndefined {
				res = fn(append([]JSValue{thisVal}, args...)...)
			} else {
				res = fn(args...)
			}
		}()
		return res, nil
	case *ObjectLiteral:
		// Evaluate object literal
		obj := make(map[string]JSValue)
		for _, prop := range e.Properties {
			// Determine key
			var key string
			switch k := prop.Key.(type) {
			case *Identifier:
				key = k.Name
			case *Literal:
				key = fmt.Sprint(k.Value)
			default:
				continue
			}
			v, err := evalExpression(prop.Value, ctx)
			if err != nil {
				return Undefined(), err
			}
			obj[key] = v
		}
		return JSValue{Type: JSObject, Object: obj}, nil
	case *ArrayLiteral:
		// Evaluate array literal
		arr := []JSValue{}
		for _, elem := range e.Elements {
			v, err := evalExpression(elem, ctx)
			if err != nil {
				return Undefined(), err
			}
			arr = append(arr, v)
		}
		return JSValue{Type: JSObject, Object: arr, IsArray: true}, nil
	case *AssignmentExpr:
		// Handle assignment a = b and compound assignments
		// Only support '=' for now
		if e.Operator != "=" {
			return Undefined(), fmt.Errorf("unsupported assignment operator: %s", e.Operator)
		}
		// Evaluate right-hand side
		val, err := evalExpression(e.Right, ctx)
		if err != nil {
			return Undefined(), err
		}
		// Assign to identifier or member expression
		switch target := e.Left.(type) {
		case *Identifier:
			ctx.global.SetInChain(target.Name, val)
		case *MemberExpr:
			// evaluate object
			objVal, err := evalExpression(target.Object, ctx)
			if err != nil {
				return Undefined(), err
			}
			m := objVal.ToObject()
			// determine key
			var key string
			if target.Computed {
				k, err2 := evalExpression(target.Property, ctx)
				if err2 != nil {
					return Undefined(), err2
				}
				key = k.ToString()
			} else if id, ok := target.Property.(*Identifier); ok {
				key = id.Name
			}
			m[key] = val
		default:
			return Undefined(), fmt.Errorf("invalid assignment target: %T", e.Left)
		}
		return val, nil
	default:
		return Undefined(), nil // TODO: 其它表达式类型
	}
}

func convertLiteral(lit *Literal) JSValue {
	switch v := lit.Value.(type) {
	case float64:
		return NumberVal(v)
	case int:
		return NumberVal(float64(v))
	case string:
		return StringVal(v)
	case bool:
		return BoolVal(v)
	case nil:
		return Null()
	default:
		return Undefined()
	}
}

func evalBinaryExpr(op string, left, right JSValue) JSValue {
	switch op {
	case "==":
		return BoolVal(left.ToString() == right.ToString())
	case "!=":
		return BoolVal(left.ToString() != right.ToString())
	case "===":
		return BoolVal(left.Type == right.Type && left.ToString() == right.ToString())
	case "!==":
		return BoolVal(left.Type != right.Type || left.ToString() != right.ToString())
	case "+":
		if left.IsNumber() && right.IsNumber() {
			return NumberVal(left.ToNumber() + right.ToNumber())
		}
		return StringVal(left.ToString() + right.ToString())
	case "-":
		return NumberVal(left.ToNumber() - right.ToNumber())
	case "*":
		return NumberVal(left.ToNumber() * right.ToNumber())
	case "/":
		return NumberVal(left.ToNumber() / right.ToNumber())
	case "<":
		return BoolVal(left.ToNumber() < right.ToNumber())
	case ">":
		return BoolVal(left.ToNumber() > right.ToNumber())
	case "<=":
		return BoolVal(left.ToNumber() <= right.ToNumber())
	case ">=":
		return BoolVal(left.ToNumber() >= right.ToNumber())
	case "&&":
		// Logical AND returns boolean result
		return BoolVal(left.ToBool() && right.ToBool())
	case "||":
		// Logical OR returns boolean result
		return BoolVal(left.ToBool() || right.ToBool())
	default:
		return Undefined()
	}
}

func evalUnaryExpr(op string, val JSValue) JSValue {
	switch op {
	case "-":
		return NumberVal(-val.ToNumber())
	case "!":
		return BoolVal(!val.ToBool())
	default:
		return val
	}
}

// try-catch 语句执行
func evalTryCatch(tc *TryCatchStmt, ctx *RunContext) (res JSValue, err error) {
	// catch JSException panics from TryBlock
	defer func() {
		if r := recover(); r != nil {
			if exc, ok := r.(*JSException); ok {
				if tc.CatchParam != nil && tc.CatchBlock != nil {
					oldScope := ctx.global
					ctx.global = NewScope(oldScope)
					ctx.global.Set(tc.CatchParam.Name, exc.Value)
					res, err = evalStatement(tc.CatchBlock, ctx)
					ctx.global = oldScope
					return
				}
			}
			// rethrow other panics and return
			panic(r)
		}
	}()
	// execute try block normally
	res, err = evalStatement(tc.TryBlock, ctx)
	return
}

type JSException struct {
	Value JSValue
}

func (e *JSException) Error() string {
	return e.Value.ToString()
}

// evalClassDecl 将 ClassDecl 编译为 constructor 与原型
func evalClassDecl(cd *ClassDecl, ctx *RunContext) (JSValue, error) {
	// 解析继承
	var superCtor JSValue = Undefined()
	var superProto JSValue = Undefined()

	if cd.SuperClass != nil {
		sc, err := evalExpression(cd.SuperClass, ctx)
		if err != nil {
			return Undefined(), err
		}
		superCtor = sc

		// 获取父类的原型
		if superCtor.Type == JSFunction && superCtor.Object != nil {
			if m, ok := superCtor.Object.(map[string]JSValue); ok {
				if sp, ok2 := m["prototype"]; ok2 {
					superProto = sp
				}
			}
		}
	}

	// 构建原型对象
	protoMap := make(map[string]JSValue)
	// 寻找 constructor 方法
	var consDecl *FunctionDecl
	for _, mdef := range cd.Body {
		md := mdef // capture
		if md.Kind == "constructor" {
			consDecl = md.Value
		} else {
			// 普通方法挂到原型
			protoMap[md.Key.Name] = FunctionVal(func(args ...JSValue) JSValue {
				// 第一个参数是this
				if len(args) == 0 {
					return Undefined()
				}
				thisVal := args[0]

				// 创建新作用域
				oldScope := ctx.global
				scope := NewScope(oldScope)
				ctx.global = scope

				// 捕获return和恢复作用域
				var result JSValue = Undefined()
				defer func() {
					ctx.global = oldScope
					if r := recover(); r != nil {
						if rp, ok := r.(ReturnPanic); ok {
							result = rp.Value
							return
						}
						panic(r)
					}
				}()

				// 绑定参数
				for i, ident := range md.Value.Params {
					if i+1 < len(args) {
						scope.Set(ident.Name, args[i+1])
					} else {
						scope.Set(ident.Name, Undefined())
					}
				}

				// 绑定this
				scope.Set("this", thisVal)

				// 创建super对象 - 直接引用父类原型
				if superProto.Type != JSUndefined {
					// 创建一个特殊的super对象，其方法调用时会绑定this为当前实例
					superObj := make(map[string]JSValue)
					protoObj := superProto.ToObject()

					// 复制父类原型的所有方法，并确保this绑定正确
					for k, v := range protoObj {
						if v.IsFunction() {
							// 创建一个闭包来捕获当前方法和this
							finalV := v // 捕获当前值
							superObj[k] = FunctionVal(func(args ...JSValue) JSValue {
								// 调用父类方法时，使用子类实例作为this
								return finalV.Function(append([]JSValue{thisVal}, args...)...)
							})
						} else {
							superObj[k] = v
						}
					}
					scope.Set("super", ObjectVal(superObj))
				} else {
					scope.Set("super", Undefined())
				}

				// 执行方法体
				if md.Value.Body != nil {
					for _, stmt := range md.Value.Body.Body {
						val, err := evalStatement(stmt, ctx)
						if err != nil {
							break
						}
						result = val
					}
				}
				return result
			})
		}
	}

	protoVal := ObjectVal(protoMap)

	// 设置原型链
	if superProto.Type != JSUndefined {
		protoVal.Proto = &superProto
	}

	// 构造 constructor
	ctor := FunctionVal(func(args ...JSValue) JSValue {
		// 创建实例并设置原型
		instMap := make(map[string]JSValue)
		inst := JSValue{Type: JSObject, Object: instMap, Proto: &protoVal}

		// 创建新作用域
		oldScope := ctx.global
		scope := NewScope(oldScope)
		ctx.global = scope

		// 捕获return和恢复作用域
		defer func() {
			ctx.global = oldScope
			if r := recover(); r != nil {
				if _, ok := r.(ReturnPanic); ok {
					return
				}
				panic(r)
			}
		}()

		// 绑定this
		scope.Set("this", inst)

		// 绑定参数
		if consDecl != nil {
			for i, ident := range consDecl.Params {
				if i < len(args) {
					scope.Set(ident.Name, args[i])
				} else {
					scope.Set(ident.Name, Undefined())
				}
			}
		}

		// 创建super对象，用于构造函数中调用super()
		if superCtor.Type != JSUndefined {
			// 在构造函数中，super是父类构造函数
			scope.Set("super", FunctionVal(func(args ...JSValue) JSValue {
				// 调用父类构造函数，传入当前实例作为this
				return superCtor.Function(append([]JSValue{inst}, args...)...)
			}))
		} else {
			scope.Set("super", Undefined())
		}

		// 执行自身 constructor
		if consDecl != nil && consDecl.Body != nil {
			for _, stmt := range consDecl.Body.Body {
				_, err := evalStatement(stmt, ctx)
				if err != nil {
					break
				}
			}
		}

		return inst
	})

	// 设置 ctor.prototype = protoVal
	if ctor.Object == nil {
		ctor.Object = make(map[string]JSValue)
	}
	ctor.Object.(map[string]JSValue)["prototype"] = protoVal
	ctx.global.Set(cd.Name.Name, ctor)
	return Undefined(), nil
}

// evalFunctionDecl 执行 FunctionDecl AST，支持 this 与 super，并捕获 return
func evalFunctionDecl(fd *FunctionDecl, thisVal JSValue, superCtor JSValue, ctx *RunContext, args ...JSValue) JSValue {
	// 建立新作用域
	oldScope := ctx.global
	scope := NewScope(oldScope)
	ctx.global = scope
	// 捕获 return panic 并恢复作用域
	var result JSValue = Undefined()
	defer func() {
		ctx.global = oldScope
		if r := recover(); r != nil {
			if rp, ok := r.(ReturnPanic); ok {
				result = rp.Value
				return
			}
			panic(r)
		}
	}()
	// 绑定参数（第一个 args 为 this）
	for i, ident := range fd.Params {
		if i+1 < len(args) {
			scope.Set(ident.Name, args[i+1])
		} else {
			scope.Set(ident.Name, Undefined())
		}
	}
	// 绑定 this
	scope.Set("this", thisVal)

	// 创建 super 对象，用于访问父类方法
	if superCtor.Type == JSFunction && superCtor.Object != nil {
		superObj := make(map[string]JSValue)
		if m, ok := superCtor.Object.(map[string]JSValue); ok {
			if superProto, ok2 := m["prototype"]; ok2 {
				// 创建一个特殊的 super 对象，其方法调用时会绑定 this 为当前实例
				protoObj := superProto.ToObject()
				for k, v := range protoObj {
					if v.IsFunction() {
						// 包装函数，确保 this 绑定正确
						superObj[k] = FunctionVal(func(args ...JSValue) JSValue {
							// 调用父类方法时，传入当前实例作为 this
							return v.Function(append([]JSValue{thisVal}, args...)...)
						})
					} else {
						superObj[k] = v
					}
				}
			}
		}
		scope.Set("super", ObjectVal(superObj))
	} else {
		scope.Set("super", Undefined())
	}

	// 执行方法体
	if fd.Body != nil {
		for _, stmt := range fd.Body.Body {
			val, err := evalStatement(stmt, ctx)
			if err != nil {
				break
			}
			result = val
		}
	}
	return result
}

// evalArguments 将 Expression 列表 Eval 到 JSValue 列表
func evalArguments(args []Expression, ctx *RunContext) []JSValue {
	var out []JSValue
	for _, arg := range args {
		v, _ := evalExpression(arg, ctx)
		out = append(out, v)
	}
	return out
}

func destructureObjectPattern(pattern *ObjectLiteral, val JSValue, ctx *RunContext) error {
	// Accept both JSObject (map[string]JSValue) and JSValue with .Object as map
	var obj map[string]JSValue
	if val.Type == JSObject {
		if m, ok := val.Object.(map[string]JSValue); ok {
			obj = m
		}
	}
	if obj == nil {
		return fmt.Errorf("object pattern expects an object value")
	}
	for _, prop := range pattern.Properties {
		// prop.Key may be Identifier or Literal
		var propName string
		switch k := prop.Key.(type) {
		case *Identifier:
			propName = k.Name
		case *Literal:
			propName = fmt.Sprint(k.Value)
		default:
			return fmt.Errorf("unsupported object pattern key: %T", prop.Key)
		}
		propVal, ok := obj[propName]
		if !ok {
			propVal = Undefined()
		}
		if err := destructurePattern(prop.Value, propVal, ctx); err != nil {
			return err
		}
	}
	return nil
}

func destructureArrayPattern(pattern *ArrayLiteral, val JSValue, ctx *RunContext) error {
	arr := val.ToArray()
	if arr == nil {
		return fmt.Errorf("array pattern expects an array value")
	}
	for i, elem := range pattern.Elements {
		if i >= len(arr) {
			elemVal := Undefined()
			if err := destructurePattern(elem, elemVal, ctx); err != nil {
				return err
			}
		} else {
			elemVal := arr[i]
			if err := destructurePattern(elem, elemVal, ctx); err != nil {
				return err
			}
		}
	}
	return nil
}

func destructurePattern(pattern Expression, val JSValue, ctx *RunContext) error {
	switch p := pattern.(type) {
	case *Identifier:
		ctx.global.SetInChain(p.Name, val)
	case *ObjectLiteral:
		return destructureObjectPattern(p, val, ctx)
	case *ArrayLiteral:
		return destructureArrayPattern(p, val, ctx)
	default:
		return fmt.Errorf("unsupported destructuring pattern: %T", pattern)
	}
	return nil
}

var ErrBreak = errors.New("break")
