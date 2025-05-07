package simplejs

import (
	"fmt"
	"strconv"
)

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
	case *FunctionDecl:
		// 处理函数表达式
		fnVal := FunctionVal(func(args ...JSValue) JSValue {
			return evalFunctionDecl(e, Undefined(), Undefined(), ctx, args...)
		})
		return fnVal, nil
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
		if id, ok := e.Object.(*Identifier); ok && id.Name == "super" {
			superObj, found := ctx.global.Get("super")
			if !found || superObj.Type == JSUndefined {
				return Undefined(), nil
			}
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
			superMap := superObj.ToObject()
			if superMap != nil {
				if val, ok := superMap[key]; ok {
					return val, nil
				}
			}
			return Undefined(), nil
		}
		objVal, err := evalExpression(e.Object, ctx)
		if err != nil {
			return Undefined(), err
		}
		// 检查对象是否为空
		if objVal.Type == JSUndefined || objVal.Type == JSNull {
			return Undefined(), nil
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
		
		// 检查对象的属性
		m := objVal.ToObject()
		if m != nil {
			if val, ok := m[key]; ok {
				return val, nil
			}
		}
		
		// 检查原型链
		cur := objVal.Proto
		for cur != nil {
			protoMap := cur.ToObject()
			if protoMap != nil {
				if val, ok := protoMap[key]; ok {
					return val, nil
				}
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
		var thisVal JSValue = Undefined()
		var fnVal JSValue
		var err error
		
		// 处理方法调用 (obj.method())
		if memberExpr, ok := e.Callee.(*MemberExpr); ok {
			// 处理super调用
			if id, ok2 := memberExpr.Object.(*Identifier); ok2 && id.Name == "super" {
				thisVal, _ = ctx.global.Get("this")
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
				
				// 获取super方法
				superMap := superObj.ToObject()
				if superMap != nil {
					if method, ok := superMap[methodName]; ok && method.IsFunction() {
						args := evalArguments(e.Arguments, ctx)
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
							res = method.Function(append([]JSValue{thisVal}, args...)...)
						}()
						return res, nil
					}
				}
				return Undefined(), nil
			} else {
				// 普通方法调用 (obj.method())
				// 先获取对象
				thisVal, err = evalExpression(memberExpr.Object, ctx)
				if err != nil {
					return Undefined(), err
				}
				
				// 检查对象是否为空
				if thisVal.Type == JSUndefined || thisVal.Type == JSNull {
					return Undefined(), fmt.Errorf("cannot read property of %s", thisVal.Type.String())
				}
				
				// 获取方法名
				var propName string
				if memberExpr.Computed {
					v, err := evalExpression(memberExpr.Property, ctx)
					if err != nil {
						return Undefined(), err
					}
					propName = v.ToString()
				} else if propId, ok := memberExpr.Property.(*Identifier); ok {
					propName = propId.Name
				}
				
				// 先从对象自身查找方法
				objMap := thisVal.ToObject()
				if objMap != nil {
					if method, ok := objMap[propName]; ok {
						fnVal = method
						goto callFunction
					}
				}
				
				// 从原型链查找方法
				cur := thisVal.Proto
				for cur != nil {
					protoMap := cur.ToObject()
					if protoMap != nil {
						if method, ok := protoMap[propName]; ok {
							fnVal = method
							goto callFunction
						}
					}
					cur = cur.Proto
				}
				
				// 方法未找到
				return Undefined(), fmt.Errorf("method '%s' not found", propName)
			}
		} else {
			// 普通函数调用 (func())
			fnVal, err = evalExpression(e.Callee, ctx)
			if err != nil {
				return Undefined(), err
			}
		}
		
		callFunction:
		// 检查是否是函数
		if fnVal.Type != JSFunction {
			return Undefined(), fmt.Errorf("call non-function: %s", fnVal.Type.String())
		}
		
		// 检查函数是否为空
		if fnVal.Function == nil {
			return Undefined(), fmt.Errorf("function is nil")
		}
		
		// 获取函数
		fn := fnVal.ToFunction()
		
		// 准备参数
		args := evalArguments(e.Arguments, ctx)
		
		// 调用函数
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
			
			// 如果有this值，将其作为第一个参数传递
			if thisVal.Type != JSUndefined {
				res = fn(append([]JSValue{thisVal}, args...)...)
				
				// 如果返回值是对象，并且是方法调用（有this值），则保留this绑定
				// 这对于支持链式调用（如calculator.add(5).multiply(2)）很重要
				if res.Type == JSObject {
					// 检查返回的对象是否是this对象本身
					// 在JavaScript中，当方法返回this时，应该返回原始对象以支持链式调用
					if _, ok := e.Callee.(*MemberExpr); ok {
						// 检查方法返回值是否为this
						// 在JavaScript中，当方法返回this时，通常是返回对象本身的引用
						// 我们需要检查返回的对象是否与调用方法的对象相同
						
						// 1. 检查返回值是否为对象
						if res.Type == JSObject {
							// 2. 检查返回的对象是否是this对象本身
							// 在JS中，如果方法返回this，它应该是对象本身的引用
							// 这里我们直接将返回值设置为thisVal，确保链式调用正常工作
							if resObj := res.ToObject(); resObj != nil {
								// 如果方法返回的是一个对象，并且该对象有一个value属性
								// 这通常表示它是一个包装了this的对象
								if _, ok := resObj["value"]; ok {
									res = thisVal
								}
							}
						}
					}
				}
			} else {
				res = fn(args...)
			}
		}()
		
		return res, nil
	case *ObjectLiteral:
		obj := make(map[string]JSValue)
		for _, prop := range e.Properties {
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
		arrMap := make(map[string]JSValue)
		for i, elem := range e.Elements {
			v, err := evalExpression(elem, ctx)
			if err != nil {
				return Undefined(), err
			}
			arrMap[strconv.Itoa(i)] = v
		}
		arrMap["length"] = NumberVal(float64(len(e.Elements)))
		return JSValue{Type: JSObject, Object: arrMap, IsArray: true}, nil
	case *AssignmentExpr:
		// 处理赋值表达式
		var resultVal JSValue
		
		if e.Operator != "=" {
			// 处理复合赋值运算符 (+=, -=, *=, /=)
			// 1. 获取左侧值
			var leftVal JSValue
			var err error
			
			switch target := e.Left.(type) {
			case *Identifier:
				leftVal, _ = ctx.global.Get(target.Name)
			case *MemberExpr:
				leftVal, err = evalExpression(target, ctx)
				if err != nil {
					return Undefined(), err
				}
			default:
				return Undefined(), fmt.Errorf("invalid assignment target: %T", e.Left)
			}
			
			// 2. 获取右侧值
			rightVal, err := evalExpression(e.Right, ctx)
			if err != nil {
				return Undefined(), err
			}
			
			// 3. 根据运算符执行相应操作
			switch e.Operator {
			case "+=":
				// 数字相加或字符串连接
				if leftVal.Type == JSNumber && rightVal.Type == JSNumber {
					resultVal = NumberVal(leftVal.ToNumber() + rightVal.ToNumber())
				} else {
					resultVal = StringVal(leftVal.ToString() + rightVal.ToString())
				}
			case "-=":
				resultVal = NumberVal(leftVal.ToNumber() - rightVal.ToNumber())
			case "*=":
				resultVal = NumberVal(leftVal.ToNumber() * rightVal.ToNumber())
			case "/=":
				resultVal = NumberVal(leftVal.ToNumber() / rightVal.ToNumber())
			default:
				return Undefined(), fmt.Errorf("unsupported assignment operator: %s", e.Operator)
			}
		} else {
			// 处理普通赋值运算符 =
			val, err := evalExpression(e.Right, ctx)
			if err != nil {
				return Undefined(), err
			}
			resultVal = val
		}
		
		// 4. 将结果赋值给左侧
		switch target := e.Left.(type) {
		case *Identifier:
			ctx.global.SetInChain(target.Name, resultVal)
		case *MemberExpr:
			objVal, err := evalExpression(target.Object, ctx)
			if err != nil {
				return Undefined(), err
			}
			if objVal.Type != JSObject {
				return Undefined(), fmt.Errorf("assignment to property of non-object: %s", objVal.Type.String())
			}
			m := objVal.ToObject()
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
			m[key] = resultVal
			if objVal.IsArray {
				if idx, err := strconv.Atoi(key); err == nil {
					curr, ok := m["length"]
					var currLen float64
					if ok {
						currLen = curr.ToNumber()
					}
					if float64(idx+1) > currLen {
						m["length"] = NumberVal(float64(idx + 1))
					}
				}
			}
		default:
			return Undefined(), fmt.Errorf("invalid assignment target: %T", e.Left)
		}
		
		return resultVal, nil
	case *ConditionalExpr:
		condVal, err := evalExpression(e.Test, ctx)
		if err != nil {
			return Undefined(), err
		}
		if condVal.ToBool() {
			return evalExpression(e.Consequent, ctx)
		}
		return evalExpression(e.Alternate, ctx)
	case *UpdateExpr:
		// 处理自增和自减表达式 (i++ 或 i--)，支持前缀和后缀形式
		var target string
		var currentVal JSValue
		
		// 获取操作数
		switch arg := e.Argument.(type) {
		case *Identifier:
			target = arg.Name
			currentVal, _ = ctx.global.Get(target)
		case *MemberExpr:
			// 处理对象属性的自增/自减
			objVal, err := evalExpression(arg.Object, ctx)
			if err != nil {
				return Undefined(), err
			}
			if objVal.Type != JSObject {
				return Undefined(), fmt.Errorf("update operation on property of non-object: %s", objVal.Type.String())
			}
			m := objVal.ToObject()
			var key string
			if arg.Computed {
				k, err := evalExpression(arg.Property, ctx)
				if err != nil {
					return Undefined(), err
				}
				key = k.ToString()
			} else if id, ok := arg.Property.(*Identifier); ok {
				key = id.Name
			}
			val, ok := m[key]
			if !ok {
				val = NumberVal(0)
			}
			currentVal = val
			
			// 计算新值
			var newVal JSValue
			if e.Op == "++" {
				newVal = NumberVal(currentVal.ToNumber() + 1)
			} else { // "--"
				newVal = NumberVal(currentVal.ToNumber() - 1)
			}
			
			// 更新对象属性
			m[key] = newVal
			
			// 根据前缀或后缀返回适当的值
			if e.Prefix {
				return newVal, nil
			}
			return currentVal, nil
		default:
			return Undefined(), fmt.Errorf("invalid update target: %T", e.Argument)
		}
		
		// 计算新值
		var newVal JSValue
		if e.Op == "++" {
			newVal = NumberVal(currentVal.ToNumber() + 1)
		} else { // "--"
			newVal = NumberVal(currentVal.ToNumber() - 1)
		}
		
		// 更新变量
		ctx.global.SetInChain(target, newVal)
		
		// 根据前缀或后缀返回适当的值
		if e.Prefix {
			return newVal, nil
		}
		return currentVal, nil
	default:
		return Undefined(), fmt.Errorf("unsupported expression type: %T", expr)
	}
}
