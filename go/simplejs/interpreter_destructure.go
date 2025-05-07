package simplejs

import (
	"strconv"
)

func destructureObjectPattern(pattern *ObjectLiteral, val JSValue, ctx *RunContext) error {
	obj := val.ToObject()
	for _, prop := range pattern.Properties {
		key := ""
		switch k := prop.Key.(type) {
		case *Identifier:
			key = k.Name
		case *Literal:
			key = k.Value.(string)
		}
		if prop.Value != nil {
			v, ok := obj[key]
			if !ok {
				v = Undefined()
			}
			if ident, ok := prop.Value.(*Identifier); ok {
				ctx.global.Set(ident.Name, v)
			}
		}
	}
	return nil
}

func destructureArrayPattern(pattern *ArrayLiteral, val JSValue, ctx *RunContext) error {
	obj := val.ToObject()
	for i, elem := range pattern.Elements {
		if elem == nil {
			continue
		}
		v, ok := obj[strconv.Itoa(i)]
		if !ok {
			v = Undefined()
		}
		if ident, ok := elem.(*Identifier); ok {
			ctx.global.Set(ident.Name, v)
		}
	}
	return nil
}

func destructurePattern(pattern Expression, val JSValue, ctx *RunContext) error {
	switch p := pattern.(type) {
	case *ObjectLiteral:
		return destructureObjectPattern(p, val, ctx)
	case *ArrayLiteral:
		return destructureArrayPattern(p, val, ctx)
	}
	return nil
}
