package sjson

import (
	"strings" // Need strings for complex JSON example
	"testing"
)

func TestLexer_NextToken(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected []Token
	}{
		{
			name:  "基本标记",
			input: "{}[]:,",
			expected: []Token{
				{Type: LeftBraceToken, Value: "{", Pos: 0},
				{Type: RightBraceToken, Value: "}", Pos: 1},
				{Type: LeftBracketToken, Value: "[", Pos: 2},
				{Type: RightBracketToken, Value: "]", Pos: 3},
				{Type: ColonToken, Value: ":", Pos: 4},
				{Type: CommaToken, Value: ",", Pos: 5},
				{Type: EOFToken, Pos: 6},
			},
		},
		{
			name:  "数字",
			input: "123 -456 7.89 1e10 -2.5e-3",
			expected: []Token{
				{Type: NumberToken, Value: "123", Pos: 0},
				{Type: NumberToken, Value: "-456", Pos: 4},
				{Type: NumberToken, Value: "7.89", Pos: 9},
				{Type: NumberToken, Value: "1e10", Pos: 14},
				{Type: NumberToken, Value: "-2.5e-3", Pos: 19},
				{Type: EOFToken, Pos: 26},
			},
		},
		{
			name:  "关键字",
			input: "true false null",
			expected: []Token{
				{Type: TrueToken, Value: "true", Pos: 0},
				{Type: FalseToken, Value: "false", Pos: 5},
				{Type: NullToken, Value: "null", Pos: 11},
				{Type: EOFToken, Pos: 15},
			},
		},
		{
			name:  "基本字符串",
			input: `"hello" "world"`,
			expected: []Token{
				{Type: StringToken, Value: "hello", Pos: 0},
				{Type: StringToken, Value: "world", Pos: 8},
				{Type: EOFToken, Pos: 15},
			},
		},
		{
			name:  "Unicode转义",
			input: `"\u0041\u4F60\u597D"`,
			expected: []Token{
				{Type: StringToken, Value: "A你好", Pos: 0},
				{Type: EOFToken, Pos: 20},
			},
		},
		{
			name:  "简单JSON对象",
			input: `{"name":"张三", "age":30}`,
			expected: []Token{
				{Type: LeftBraceToken, Value: "{", Pos: 0},
				{Type: StringToken, Value: "name", Pos: 1},
				{Type: ColonToken, Value: ":", Pos: 7},
				{Type: StringToken, Value: "张三", Pos: 8},     // "张三" is 6 bytes in UTF-8
				{Type: CommaToken, Value: ",", Pos: 16},      // Position after "张三"
				{Type: StringToken, Value: "age", Pos: 18},   // Position after ", "
				{Type: ColonToken, Value: ":", Pos: 23},      // Position after "age"
				{Type: NumberToken, Value: "30", Pos: 24},    // Position after ":"
				{Type: RightBraceToken, Value: "}", Pos: 26}, // Position after "30"
				{Type: EOFToken, Pos: 27},                    // Position after "}"
			},
		},
	}

	for _, tt := range tests {
		testCase := tt // 避免闭包问题
		t.Run(testCase.name, func(t *testing.T) {
			lexer := NewLexer(testCase.input)
			for i, expected := range testCase.expected {
				got := lexer.NextToken()
				if got.Type != expected.Type {
					t.Errorf("标记 #%d 类型错误: 期望 %v, 得到 %v", i, expected.Type, got.Type)
				}
				if got.Value != expected.Value {
					t.Errorf("标记 #%d 值错误: 期望 %q, 得到 %q", i, expected.Value, got.Value)
				}
				if got.Pos != expected.Pos {
					t.Errorf("标记 #%d 位置错误: 期望 %d, 得到 %d", i, expected.Pos, got.Pos)
				}
			}
		})
	}
}

// Benchmark for simple JSON input
func BenchmarkLexerSimple(b *testing.B) {
	input := `{
		"name": "张三",
		"age": 30,
		"city": "北京",
		"active": true,
		"scores": [100, 95, 88]
	}`
	for i := 0; i < b.N; i++ {
		lexer := NewLexer(input)
		for {
			token := lexer.NextToken()
			if token.Type == EOFToken {
				break
			}
		}
	}
}

// Benchmark for a slightly more complex JSON input
func BenchmarkLexerComplex(b *testing.B) {
	// Generate a larger JSON string for more realistic benchmarking
	var sb strings.Builder
	sb.WriteString(`{
		"widget": {
			"debug": "on",
			"window": {
				"title": "Sample Konfabulator Widget",
				"name": "main_window",
				"width": 500,
				"height": 500
			},
			"image": {
				"src": "Images/Sun.png",
				"name": "sun1",
				"hOffset": 250,
				"vOffset": 250,
				"alignment": "center"
			},
			"text": {
				"data": "Click Here",
				"size": 36,
				"style": "bold",
				"name": "text1",
				"hOffset": 250,
				"vOffset": 100,
				"alignment": "center",
				"onMouseUp": "sun1.opacity = (sun1.opacity / 100) * 90;"
			}
		}
	}`)
	input := sb.String()

	b.ResetTimer() // Reset timer after setup
	for i := 0; i < b.N; i++ {
		lexer := NewLexer(input)
		for {
			token := lexer.NextToken()
			if token.Type == EOFToken {
				break
			}
		}
	}
}

// Benchmark focusing on string parsing with escapes
func BenchmarkLexerStringEscapes(b *testing.B) {
	input := `{"escapes": "Hello\nWorld\tTab\"Quote\\Backslash\u0041\u4F60\u597D"}`
	for i := 0; i < b.N; i++ {
		lexer := NewLexer(input)
		for {
			token := lexer.NextToken()
			if token.Type == EOFToken {
				break
			}
		}
	}
}

// BenchmarkLexerAllTokens 测试包含所有token类型的复杂JSON解析性能
func BenchmarkLexerAllTokens(b *testing.B) {
	// 构建包含所有类型Token的JSON
	jsonStr := `{
		"nullValue": null,
		"boolValues": {
			"trueValue": true,
			"falseValue": false
		},
		"numbers": [
			0,
			42,
			-73,
			3.14159,
			-2.71828,
			1.23e+6,
			-4.56e-3
		],
		"strings": [
			"",
			"Hello, World!",
			"特殊字符: \\, \", \/, \b, \f, \n, \r, \t",
			"Unicode: \u0041\u4F60\u597D"
		],
		"nestedObjects": {
			"level1": {
				"level2": {
					"level3": {
						"deep": "nested value"
					},
					"array": [1, 2, [3, 4, [5]]]
				}
			}
		},
		"mixedArray": [
			null,
			true,
			false,
			123,
			-45.67,
			"string",
			[1, 2, 3],
			{"key": "value"}
		],
		"emptyStructures": {
			"emptyObject": {},
			"emptyArray": []
		}
	}`

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		lexer := NewLexer(jsonStr)
		for {
			token := lexer.NextToken()
			if token.Type == EOFToken {
				break
			}
		}
	}
}

// BenchmarkLexerBatchProcessing 测试批量处理性能
func BenchmarkLexerBatchProcessing(b *testing.B) {
	// 使用相同的复杂JSON
	jsonStr := `{
		"nullValue": null,
		"boolValues": {
			"trueValue": true,
			"falseValue": false
		},
		"numbers": [
			0,
			42,
			-73,
			3.14159,
			-2.71828,
			1.23e+6,
			-4.56e-3
		],
		"strings": [
			"",
			"Hello, World!",
			"特殊字符: \\, \", \/, \b, \f, \n, \r, \t",
			"Unicode: \u0041\u4F60\u597D"
		],
		"nestedObjects": {
			"level1": {
				"level2": {
					"level3": {
						"deep": "nested value"
					},
					"array": [1, 2, [3, 4, [5]]]
				}
			}
		},
		"mixedArray": [
			null,
			true,
			false,
			123,
			-45.67,
			"string",
			[1, 2, 3],
			{"key": "value"}
		],
		"emptyStructures": {
			"emptyObject": {},
			"emptyArray": []
		}
	}`

	b.Run("单个标记处理", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			lexer := NewLexer(jsonStr)
			for {
				token := lexer.NextToken()
				if token.Type == EOFToken {
					break
				}
			}
		}
	})

	b.Run("批量标记处理", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			lexer := NewLexer(jsonStr)
			tokens := lexer.ParseAll()
			_ = tokens
		}
	})
}
