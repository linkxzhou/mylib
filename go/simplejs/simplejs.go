package simplejs

import "fmt"

// SimpleJS is the top-level JS engine.
type SimpleJS struct {
	ctx *RunContext
}

// NewSimpleJS creates a new SimpleJS instance.
func NewSimpleJS(size int) *SimpleJS {
	return &SimpleJS{ctx: NewContext(size)}
}

// Eval evaluates JS code.
func (s *SimpleJS) Eval(code string) (JSValue, error) {
	return s.ctx.Eval(code)
}

// ToString converts a JSValue to string.
func (s *SimpleJS) ToString(v JSValue) string {
	switch v.Type {
	case JSUndefined:
		return "undefined"
	case JSNull:
		return "null"
	case JSBoolean:
		if v.Bool {
			return "true"
		}
		return "false"
	case JSNumber:
		return fmt.Sprintf("%v", v.Number)
	case JSString:
		return v.String
	case JSObject:
		return "[object Object]"
	case JSFunction:
		return "[Function]"
	case JSError:
		return v.Error.Error()
	default:
		return ""
	}
}

// SetMaxCss and SetGCThreshold are no-ops in this Go version.
func (s *SimpleJS) SetMaxCss(css int)      {}
func (s *SimpleJS) SetGCThreshold(gct int) {}
