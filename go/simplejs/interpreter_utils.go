package simplejs

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
		return BoolVal(left.ToBool() && right.ToBool())
	case "||":
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

func evalArguments(args []Expression, ctx *RunContext) []JSValue {
	vals := make([]JSValue, len(args))
	for i, arg := range args {
		v, _ := evalExpression(arg, ctx)
		vals[i] = v
	}
	return vals
}
