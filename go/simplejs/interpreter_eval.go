package simplejs

import (
	"fmt"
	"log/slog"
)

// Statement evaluation
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
		if s.Init != nil {
			_, err := evalStatement(s.Init, ctx)
			if err != nil && err != ErrBreak {
				return Undefined(), err
			}
		}
		var result JSValue = Undefined()
		for {
			if s.Test != nil {
				cond, err := evalExpression(s.Test, ctx)
				if err != nil {
					return Undefined(), err
				}
				if !cond.ToBool() {
					break
				}
			}
			val, err := evalStatement(s.Body, ctx)
			if err == ErrBreak {
				break
			}
			if err != nil {
				return val, err
			}
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
		panic(&JSException{Value: arg})
	case *BreakStmt:
		return Undefined(), ErrBreak
	case *ExpressionStmt:
		if fnDecl, ok := s.Expr.(*FunctionDecl); ok {
			return evalStatement(fnDecl, ctx)
		}
		return evalExpression(s.Expr, ctx)
	case *ClassDecl:
		return evalClassDecl(s, ctx)
	case *FunctionDecl:
		fnVal := FunctionVal(func(args ...JSValue) JSValue {
			return evalFunctionDecl(s, Undefined(), Undefined(), ctx, args...)
		})
		if s.Name != nil {
			slog.Info("[evalStatement] Register FunctionDecl", "name", s.Name.Name)
			ctx.global.Root().Set(s.Name.Name, fnVal)
		} else {
			slog.Info("[evalStatement] FunctionDecl with nil Name, not registered")
		}
		return Undefined(), nil
	default:
		return Undefined(), nil
	}
}
