package simplejs

// RunContext holds execution state.
type RunContext struct {
	mem         *Memory
	global      *Scope
	sourceLines *[]string
}

// NewContext creates a JS runtime context.
func NewContext(size int) *RunContext {
	return &RunContext{mem: NewMemory(size), global: NewScope(nil)}
}

// Eval executes code: lex, parse, and run statements.
func (ctx *RunContext) Eval(code string) (JSValue, error) {
	tokens, err := Tokenize(code)
	if err != nil {
		return Undefined(), err
	}
	lines := splitLines(code)
	ctx.sourceLines = &lines
	parser := NewParser(tokens, ctx)
	var res JSValue
	for parser.peek().Type != TokEOF {
		res, err = parser.ParseStatement()
		if err != nil {
			return res, err
		}
	}
	return res, nil
}

// GC triggers garbage collection (optional in Go).
func (ctx *RunContext) GC() {
	// no-op, rely on Go GC
}

// RegisterGoFunc registers a Go function into the JS global scope.
func (ctx *RunContext) RegisterGoFunc(name string, fn func(args ...JSValue) JSValue) {
	ctx.global.Set(name, FunctionVal(fn))
}

// splitLines splits code into lines for debug
func splitLines(code string) []string {
	return splitBy(code, '\n')
}

func splitBy(s string, sep rune) []string {
	var lines []string
	line := ""
	for _, r := range s {
		if r == sep {
			lines = append(lines, line)
			line = ""
		} else {
			line += string(r)
		}
	}
	lines = append(lines, line)
	return lines
}
