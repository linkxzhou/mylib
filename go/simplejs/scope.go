package simplejs

// Scope represents a lexical scope with parent chain.
type Scope struct {
	vars   map[string]JSValue
	parent *Scope
}

// NewScope creates a new scope with optional parent.
func NewScope(parent *Scope) *Scope {
	return &Scope{vars: make(map[string]JSValue), parent: parent}
}

// Get looks up a variable in the scope chain.
func (s *Scope) Get(name string) (JSValue, bool) {
	if val, ok := s.vars[name]; ok {
		return val, true
	}
	if s.parent != nil {
		return s.parent.Get(name)
	}
	return Undefined(), false
}

// Set sets a variable in the current scope.
func (s *Scope) Set(name string, val JSValue) {
	s.vars[name] = val
}

// SetInChain sets a variable in the nearest scope where it exists, or in current if not found.
func (s *Scope) SetInChain(name string, val JSValue) {
	if _, ok := s.vars[name]; ok {
		s.vars[name] = val
		return
	}
	if s.parent != nil {
		s.parent.SetInChain(name, val)
		return
	}
	s.vars[name] = val
}

// Delete removes a variable from the current scope.
// Returns true if the variable was found and deleted, false otherwise.
func (s *Scope) Delete(name string) bool {
	if _, ok := s.vars[name]; ok {
		delete(s.vars, name)
		return true
	}
	return false
}
