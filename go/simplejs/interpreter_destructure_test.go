package simplejs

import (
	"testing"
)

func TestDestructureObjectPattern(t *testing.T) {
	tests := []struct {
		name     string
		pattern  *ObjectLiteral
		value    JSValue
		expected map[string]JSValue
	}{
		{
			name: "Basic Object Destructuring",
			pattern: &ObjectLiteral{
				Properties: []*Property{
					{
						Key:   &Identifier{Name: "x"},
						Value: &Identifier{Name: "a"},
					},
					{
						Key:   &Identifier{Name: "y"},
						Value: &Identifier{Name: "b"},
					},
				},
			},
			value: ObjectVal(map[string]JSValue{
				"x": NumberVal(1),
				"y": NumberVal(2),
			}),
			expected: map[string]JSValue{
				"a": NumberVal(1),
				"b": NumberVal(2),
			},
		},
		{
			name: "String Key Object Destructuring",
			pattern: &ObjectLiteral{
				Properties: []*Property{
					{
						Key:   &Literal{Value: "x"},
						Value: &Identifier{Name: "a"},
					},
				},
			},
			value: ObjectVal(map[string]JSValue{
				"x": StringVal("hello"),
			}),
			expected: map[string]JSValue{
				"a": StringVal("hello"),
			},
		},
		{
			name: "Return Undefined When Property Missing",
			pattern: &ObjectLiteral{
				Properties: []*Property{
					{
						Key:   &Identifier{Name: "missing"},
						Value: &Identifier{Name: "c"},
					},
				},
			},
			value: ObjectVal(map[string]JSValue{}),
			expected: map[string]JSValue{
				"c": Undefined(),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := NewContext(1024)
			err := destructureObjectPattern(tt.pattern, tt.value, ctx)
			if err != nil {
				t.Errorf("destructureObjectPattern() error = %v", err)
				return
			}

			// Verify destructuring results
			for name, expected := range tt.expected {
				actual, ok := ctx.global.Get(name)
				if !ok {
					t.Errorf("Variable %s not found in global scope", name)
					continue
				}

				if actual.Type != expected.Type {
					t.Errorf("Variable %s type mismatch: expected %v, got %v", name, expected.Type, actual.Type)
					continue
				}

				switch actual.Type {
				case JSNumber:
					if actual.Number != expected.Number {
						t.Errorf("Variable %s value mismatch: expected %v, got %v", name, expected.Number, actual.Number)
					}
				case JSString:
					if actual.String != expected.String {
						t.Errorf("Variable %s value mismatch: expected %v, got %v", name, expected.String, actual.String)
					}
				}
			}
		})
	}
}

func TestDestructureArrayPattern(t *testing.T) {
	tests := []struct {
		name     string
		pattern  *ArrayLiteral
		value    JSValue
		expected map[string]JSValue
	}{
		{
			name: "Basic Array Destructuring",
			pattern: &ArrayLiteral{
				Elements: []Expression{
					&Identifier{Name: "a"},
					&Identifier{Name: "b"},
					&Identifier{Name: "c"},
				},
			},
			value: ObjectVal(map[string]JSValue{
				"0": NumberVal(1),
				"1": NumberVal(2),
				"2": NumberVal(3),
			}),
			expected: map[string]JSValue{
				"a": NumberVal(1),
				"b": NumberVal(2),
				"c": NumberVal(3),
			},
		},
		{
			name: "Array Destructuring Skip Elements",
			pattern: &ArrayLiteral{
				Elements: []Expression{
					&Identifier{Name: "a"},
					nil,
					&Identifier{Name: "c"},
				},
			},
			value: ObjectVal(map[string]JSValue{
				"0": NumberVal(1),
				"1": NumberVal(2),
				"2": NumberVal(3),
			}),
			expected: map[string]JSValue{
				"a": NumberVal(1),
				"c": NumberVal(3),
			},
		},
		{
			name: "Return Undefined When Array Element Missing",
			pattern: &ArrayLiteral{
				Elements: []Expression{
					&Identifier{Name: "a"},
					&Identifier{Name: "b"},
					&Identifier{Name: "c"},
				},
			},
			value: ObjectVal(map[string]JSValue{
				"0": NumberVal(1),
			}),
			expected: map[string]JSValue{
				"a": NumberVal(1),
				"b": Undefined(),
				"c": Undefined(),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := NewContext(1024)
			err := destructureArrayPattern(tt.pattern, tt.value, ctx)
			if err != nil {
				t.Errorf("destructureArrayPattern() error = %v", err)
				return
			}

			// Verify destructuring results
			for name, expected := range tt.expected {
				actual, ok := ctx.global.Get(name)
				if !ok {
					t.Errorf("Variable %s not found in global scope", name)
					continue
				}

				if actual.Type != expected.Type {
					t.Errorf("Variable %s type mismatch: expected %v, got %v", name, expected.Type, actual.Type)
					continue
				}

				switch actual.Type {
				case JSNumber:
					if actual.Number != expected.Number {
						t.Errorf("Variable %s value mismatch: expected %v, got %v", name, expected.Number, actual.Number)
					}
				}
			}
		})
	}
}

func TestDestructurePattern(t *testing.T) {
	tests := []struct {
		name     string
		pattern  Expression
		value    JSValue
		expected map[string]JSValue
	}{
		{
			name: "Object Destructuring Pattern",
			pattern: &ObjectLiteral{
				Properties: []*Property{
					{
						Key:   &Identifier{Name: "x"},
						Value: &Identifier{Name: "a"},
					},
				},
			},
			value: ObjectVal(map[string]JSValue{
				"x": NumberVal(42),
			}),
			expected: map[string]JSValue{
				"a": NumberVal(42),
			},
		},
		{
			name: "Array Destructuring Pattern",
			pattern: &ArrayLiteral{
				Elements: []Expression{
					&Identifier{Name: "a"},
					&Identifier{Name: "b"},
				},
			},
			value: ObjectVal(map[string]JSValue{
				"0": NumberVal(1),
				"1": NumberVal(2),
			}),
			expected: map[string]JSValue{
				"a": NumberVal(1),
				"b": NumberVal(2),
			},
		},
		{
			name: "Non-Destructuring Pattern",
			pattern: &Identifier{Name: "notPattern"},
			value:   NumberVal(42),
			expected: map[string]JSValue{
				// No variables should be set
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := NewContext(1024)
			err := destructurePattern(tt.pattern, tt.value, ctx)
			if err != nil {
				t.Errorf("destructurePattern() error = %v", err)
				return
			}

			// Verify destructuring results
			for name, expected := range tt.expected {
				actual, ok := ctx.global.Get(name)
				if !ok {
					t.Errorf("Variable %s not found in global scope", name)
					continue
				}

				if actual.Type != expected.Type {
					t.Errorf("Variable %s type mismatch: expected %v, got %v", name, expected.Type, actual.Type)
					continue
				}

				switch actual.Type {
				case JSNumber:
					if actual.Number != expected.Number {
						t.Errorf("Variable %s value mismatch: expected %v, got %v", name, expected.Number, actual.Number)
					}
				}
			}

			// For non-destructuring pattern, ensure no variables are set
			if _, ok := tt.pattern.(*Identifier); ok {
				if len(tt.expected) == 0 {
					// Check if notPattern variable was incorrectly set
					if _, ok := ctx.global.Get("notPattern"); ok {
						t.Errorf("Non-destructuring pattern should not set any variables, but 'notPattern' was set")
					}
				}
			}
		})
	}
}