package simplejs

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
