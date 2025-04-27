package simplejs

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

// SetMaxCss and SetGCThreshold are no-ops in this Go version.
func (s *SimpleJS) SetMaxCss(css int)      {}
func (s *SimpleJS) SetGCThreshold(gct int) {}
