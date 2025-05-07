package simplejs

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

// Helper function: execute code and return result
func evalTestCode(code string) (JSValue, error) {
	js := NewSimpleJS(1024 * 1024)
	return js.Eval(code)
}

// Test variable declaration statements
func TestEvalVarDeclStatement(t *testing.T) {
	tests := []struct {
		name     string
		code     string
		expected interface{}
	}{
		{
			name:     "Basic variable declaration",
			code:     "var x = 10; x;",
			expected: float64(10),
		},
		{
			name:     "Object destructuring assignment",
			code:     "var {a, b} = {a: 1, b: 2}; a + b;",
			expected: float64(3),
		},
		{
			name:     "Array destructuring assignment",
			code:     "var [x, y] = [5, 10]; x * y;",
			expected: float64(50),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := evalTestCode(tt.code)
			assert.NoError(t, err)
			switch expected := tt.expected.(type) {
			case float64:
				assert.Equal(t, expected, result.ToNumber())
			case string:
				assert.Equal(t, expected, result.ToString())
			case bool:
				assert.Equal(t, expected, result.ToBool())
			}
		})
	}
}

// Test block statements
func TestEvalBlockStatement(t *testing.T) {
	tests := []struct {
		name     string
		code     string
		expected interface{}
	}{
		{
			name:     "Simple block",
			code:     "{ var x = 5; x = x * 2; x; }",
			expected: float64(10),
		},
		{
			name:     "Nested blocks",
			code:     "{ var x = 1; { var y = 2; x = x + y; } x; }",
			expected: float64(3),
		},
		{
			name:     "Block scope",
			code:     "var x = 1; { var x = 2; } x;",
			expected: float64(1),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := evalTestCode(tt.code)
			assert.NoError(t, err)
			assert.Equal(t, tt.expected, result.ToNumber())
		})
	}
}

// Test conditional statements
func TestEvalIfStatement(t *testing.T) {
	tests := []struct {
		name     string
		code     string
		expected interface{}
	}{
		{
			name:     "If condition is true",
			code:     "var x = 0; if (true) { x = 1; } x;",
			expected: float64(1),
		},
		{
			name:     "If condition is false",
			code:     "var x = 0; if (false) { x = 1; } x;",
			expected: float64(0),
		},
		{
			name:     "If-else statement",
			code:     "var x = 0; if (false) { x = 1; } else { x = 2; } x;",
			expected: float64(2),
		},
		{
			name:     "If-else if-else statement",
			code:     "var x = 0; if (false) { x = 1; } else if (true) { x = 2; } else { x = 3; } x;",
			expected: float64(2),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := evalTestCode(tt.code)
			assert.NoError(t, err)
			assert.Equal(t, tt.expected, result.ToNumber())
		})
	}
}

// Test loop statements
func TestEvalLoopStatements(t *testing.T) {
	tests := []struct {
		name     string
		code     string
		expected interface{}
	}{
		{
			name:     "While loop",
			code:     "var i = 0; var sum = 0; while (i < 5) { sum += i; i++; } sum;",
			expected: float64(10), // 0+1+2+3+4
		},
		{
			name:     "For loop",
			code:     "var sum = 0; for (var i = 0; i < 5; i++) { sum += i; } sum;",
			expected: float64(10), // 0+1+2+3+4
		},
		{
			name:     "Break statement",
			code:     "var i = 0; var sum = 0; while (i < 10) { if (i >= 5) { break; } sum += i; i++; } sum;",
			expected: float64(10), // 0+1+2+3+4
		},
		{
			name:     "Break in for loop",
			code:     "var sum = 0; for (var i = 0; i < 10; i++) { if (i >= 5) { break; } sum += i; } sum;",
			expected: float64(10), // 0+1+2+3+4
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := evalTestCode(tt.code)
			assert.NoError(t, err)
			assert.Equal(t, tt.expected, result.ToNumber())
		})
	}
}

// Test function declaration and invocation
func TestEvalFunctionDeclaration(t *testing.T) {
	tests := []struct {
		name     string
		code     string
		expected interface{}
	}{
		{
			name:     "Simple function declaration and call",
			code:     "function add(a, b) { return a + b; } add(2, 3);",
			expected: float64(5),
		},
		{
			name:     "Function scope",
			code:     "var x = 1; function test() { var x = 2; return x; } test() + x;",
			expected: float64(3), // 2 + 1
		},
		{
			name:     "Function expression",
			code:     "var multiply = function(a, b) { return a * b; }; multiply(4, 5);",
			expected: float64(20),
		},
		{
			name:     "Recursive function",
			code:     "function factorial(n) { if (n <= 1) return 1; return n * factorial(n-1); } factorial(5);",
			expected: float64(120), // 5! = 120
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := evalTestCode(tt.code)
			assert.NoError(t, err)
			assert.Equal(t, tt.expected, result.ToNumber())
		})
	}
}

// Test class declaration and usage
func TestEvalClassDeclaration(t *testing.T) {
	tests := []struct {
		name     string
		code     string
		expected interface{}
	}{
		{
			name: "Basic class declaration and instantiation",
			code: `
			class Person {
				constructor(name) {
					this.name = name;
				}
				getName() {
					return this.name;
				}
			}
			var p = new Person("Alice");
			p.getName();
			`,
			expected: "Alice",
		},
		{
			name: "Class inheritance",
			code: `
			class Animal {
				constructor(name) {
					this.name = name;
				}
				speak() {
					return "Animal speaks";
				}
			}
			class Dog extends Animal {
				speak() {
					return this.name + " barks";
				}
			}
			var dog = new Dog("Rex");
			dog.speak();
			`,
			expected: "Rex barks",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := evalTestCode(tt.code)
			assert.NoError(t, err)
			assert.Equal(t, tt.expected, result.ToString())
		})
	}
}

// Test exception handling
func TestEvalTryCatchStatement(t *testing.T) {
	tests := []struct {
		name     string
		code     string
		expected interface{}
	}{
		{
			name: "Basic try-catch",
			code: `
			var result = "";
			try {
				throw "Error occurred";
				result = "No error";
			} catch (e) {
				result = "Caught: " + e;
			}
			result;
			`,
			expected: "Caught: Error occurred",
		},
		{
			name: "Try-catch with no exception",
			code: `
			var result = "";
			try {
				result = "No error";
			} catch (e) {
				result = "Caught: " + e;
			}
			result;
			`,
			expected: "No error",
		},
		{
			name: "Nested try-catch",
			code: `
			var result = "";
			try {
				try {
					throw "Inner error";
				} catch (e) {
					result = "Inner caught: " + e;
					throw "Outer error";
				}
			} catch (e) {
				result += ", Outer caught: " + e;
			}
			result;
			`,
			expected: "Inner caught: Inner error, Outer caught: Outer error",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := evalTestCode(tt.code)
			assert.NoError(t, err)
			assert.Equal(t, tt.expected, result.ToString())
		})
	}
}

// Test expression statements
func TestEvalExpressionStatement(t *testing.T) {
	tests := []struct {
		name     string
		code     string
		expected interface{}
	}{
		{
			name:     "Simple expression",
			code:     "5 + 3 * 2;",
			expected: float64(11),
		},
		{
			name:     "Assignment expression",
			code:     "var x = 10; x = x + 5; x;",
			expected: float64(15),
		},
		{
			name:     "Function call expression",
			code:     "function greet() { return 'Hello'; } greet();",
			expected: "Hello",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := evalTestCode(tt.code)
			assert.NoError(t, err)
			switch expected := tt.expected.(type) {
			case float64:
				assert.Equal(t, expected, result.ToNumber())
			case string:
				assert.Equal(t, expected, result.ToString())
			}
		})
	}
}

// Test complex scenarios
func TestEvalComplexScenarios(t *testing.T) {
	tests := []struct {
		name     string
		code     string
		expected interface{}
	}{
		{
			name: "Fibonacci sequence",
			code: `
			function fibonacci(n) {
				if (n <= 1) return n;
				return fibonacci(n-1) + fibonacci(n-2);
			}
			fibonacci(6);
			`,
			expected: float64(8), // fib(6) = 8
		},
		{
			name: "Closure",
			code: `
			function makeCounter() {
				var count = 0;
				return function() {
					return count++;
				};
			}
			var counter = makeCounter();
			counter(); // 0
			counter(); // 1
			counter(); // 2
			`,
			expected: float64(2),
		},
		{
			name: "Object methods and this",
			code: `
			var calculator = {
				value: 0,
				add: function(x) {
					this.value += x;
					return this;
				},
				multiply: function(x) {
					this.value *= x;
					return this;
				},
				getValue: function() {
					return this.value;
				}
			};
			calculator.add(5).multiply(2).add(10).getValue();
			`,
			expected: float64(20), // (0+5)*2+10 = 20
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := evalTestCode(tt.code)
			assert.NoError(t, err)
			assert.Equal(t, tt.expected, result.ToNumber())
		})
	}
}