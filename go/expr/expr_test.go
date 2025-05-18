package expr

import (
	"fmt"
	"strings"
	"testing"
)

var builtin = map[string]BuiltinFn{
	"sum": func(args ...interface{}) (interface{}, error) {
		var sum int64
		for _, v := range args {
			if v1, ok := v.(int64); !ok {
				return nil, fmt.Errorf("%v int64 invalid", v)
			} else {
				sum += v1
			}
		}
		return sum, nil
	},
}

func TestExpr1(t *testing.T) {
	pool, _ := NewPool(WithBuiltinList(builtin))

	tests := []struct {
		name    string
		evalstr string
		args    map[string]interface{}
		want    string
		wantErr bool
	}{
		{
			name:    `case_sum`,
			evalstr: `(sum(1,2,3) + x + y + x*y + x/y) || (1+0)`,
			args: map[string]interface{}{
				"x": 122.0,
				"y": 123.0,
			},
			want:    `true`,
			wantErr: false,
		},
		{
			name:    `case_add`,
			evalstr: `1 + 2 - 3 + 4 - 5 + 6`,
			args:    nil,
			want:    `5`,
			wantErr: false,
		},
		{
			name:    `case1_mul`,
			evalstr: `6/6 + 6/0`,
			args:    nil,
			want:    `1`,
			wantErr: true,
		},
		{
			name:    `case2_mul`,
			evalstr: `6.0/6.0`,
			args:    nil,
			want:    `1`,
			wantErr: false,
		},
		{
			name:    `case1_float`,
			evalstr: `6.12 + 6.34`,
			args:    nil,
			want:    `12.46`,
			wantErr: false,
		},
		{
			name:    `case_string`,
			evalstr: `"6.12" + "0.1111"`,
			args:    nil,
			want:    `6.120.1111`,
			wantErr: false,
		},
		{
			name:    `exp1`,
			evalstr: `5 + 2 * (3 + 1)`,
			args:    nil,
			want:    `13`,
			wantErr: false,
		},
		{
			name:    `exp2`,
			evalstr: `10 % 3`,
			args:    nil,
			want:    `1`,
			wantErr: false,
		},
		{
			name:    `exp3`,
			evalstr: `10 / 3`,
			args:    nil,
			want:    `3`,
			wantErr: false,
		},
		{
			name:    `exp4`,
			evalstr: `10.0 / 3.0`,
			args:    nil,
			want:    `3.3333333333333335`,
			wantErr: false,
		},
		{
			name:    `exp5`,
			evalstr: `2 << 3`,
			args:    nil,
			want:    `16`,
			wantErr: false,
		},
		{
			name:    `exp6`,
			evalstr: `16 >> 3`,
			args:    nil,
			want:    `2`,
			wantErr: false,
		},
		{
			name:    `exp7`,
			evalstr: `true && false`,
			args:    nil,
			want:    `false`,
			wantErr: false,
		},
		{
			name:    `exp8`,
			evalstr: `true || false`,
			args:    nil,
			want:    `true`,
			wantErr: false,
		},
		{
			name:    `exp9`,
			evalstr: `5 & 3`,
			args:    nil,
			want:    `1`,
			wantErr: false,
		},
		{
			name:    `exp10`,
			evalstr: `5 | 3`,
			args:    nil,
			want:    `7`,
			wantErr: false,
		},
		{
			name:    `exp11`,
			evalstr: `"Hello " + "World"`,
			args:    nil,
			want:    `Hello World`,
			wantErr: false,
		},
		{
			name:    `exp12`,
			evalstr: `5 - 3 * 2`,
			args:    nil,
			want:    `-1`,
			wantErr: false,
		},
		{
			name:    `exp13`,
			evalstr: `(5 - 3) * 2`,
			args:    nil,
			want:    `4`,
			wantErr: false,
		},
		{
			name:    `exp14`,
			evalstr: `10 / (5 - 3)`,
			args:    nil,
			want:    `5`,
			wantErr: false,
		},
		{
			name:    `exp15`,
			evalstr: `10 % (5 - 3)`,
			args:    nil,
			want:    `0`,
			wantErr: false,
		},
		{
			name:    `exp16`,
			evalstr: `2 << (3 + 1)`,
			args:    nil,
			want:    `32`,
			wantErr: false,
		},
		{
			name:    `exp17`,
			evalstr: `(16 >> 3) + 2`,
			args:    nil,
			want:    `4`,
			wantErr: false,
		},
		{
			name:    `exp18`,
			evalstr: `(true && false) || true`,
			args:    nil,
			want:    `true`,
			wantErr: false,
		},
		{
			name:    `exp19`,
			evalstr: `(5 & 3) * 2`,
			args:    nil,
			want:    `2`,
			wantErr: false,
		},
		{
			name:    `exp20`,
			evalstr: `("Hello " + "World") + "!"`,
			args:    nil,
			want:    `Hello World!`,
			wantErr: false,
		},
		{
			name:    `expComplex`,
			evalstr: `((2 << 3) + (10 % 3)) * (5 - (x * 2)) + (3.0 / y) * (2.0 + 1.0) && ((z + "World") == "Hello World")`,
			args: map[string]interface{}{
				"x": 3.0,
				"y": 2.0,
				"z": "Hello ",
			},
			want:    `true`,
			wantErr: false,
		},
		{
			name:    `neg`,
			evalstr: `-5 + 2`,
			args:    nil,
			want:    `-3`,
			wantErr: false,
		},
		{
			name:    `eq_true`,
			evalstr: `5 == 5`,
			args:    nil,
			want:    `true`,
			wantErr: false,
		},
		{
			name:    `ne_false`,
			evalstr: `5 != 5`,
			args:    nil,
			want:    `false`,
			wantErr: false,
		},
		{
			name:    `raw_eq`,
			evalstr: `"hello" == "hello"`,
			args:    nil,
			want:    `true`,
			wantErr: false,
		},
		{
			name:    `sum_no_args`,
			evalstr: `sum()`,
			args:    nil,
			want:    `0`,
			wantErr: false,
		},
		{
			name:    `sum_times`,
			evalstr: `sum(1,2,3) * 2`,
			args:    nil,
			want:    `12`,
			wantErr: false,
		},
		{
			name:    `not_bool`,
			evalstr: `!false`,
			args:    nil,
			want:    `true`,
			wantErr: false,
		},
		{
			name:    `concat_eq`,
			evalstr: `("abc" + "" + "def") == "abcdef"`,
			args:    nil,
			want:    `true`,
			wantErr: false,
		},
		{
			name:    `and_bool`,
			evalstr: `true && false`,
			args:    nil,
			want:    `false`,
			wantErr: false,
		},
		{
			name:    `or_bool`,
			evalstr: `true || false`,
			args:    nil,
			want:    `true`,
			wantErr: false,
		},
		{
			name:    `bit_and`,
			evalstr: `5 & 2`,
			args:    nil,
			want:    `0`,
			wantErr: false,
		},
		{
			name:    `bit_or`,
			evalstr: `5 | 2`,
			args:    nil,
			want:    `7`,
			wantErr: false,
		},
		{
			name:    `bit_xor`,
			evalstr: `5 ^ 2`,
			args:    nil,
			want:    `7`,
			wantErr: false,
		},
		{
			name:    `shl`,
			evalstr: `1 << 4`,
			args:    nil,
			want:    `16`,
			wantErr: false,
		},
		{
			name:    `shr`,
			evalstr: `16 >> 2`,
			args:    nil,
			want:    `4`,
			wantErr: false,
		},
		{
			name:    `div`,
			evalstr: `10 / 2`,
			args:    nil,
			want:    `5`,
			wantErr: false,
		},
		{
			name:    `div_zero`,
			evalstr: `1 / 0`,
			args:    nil,
			wantErr: true,
		},
		{
			name:    `type_mismatch`,
			evalstr: `1 + "a"`,
			args:    nil,
			wantErr: true,
		},
		// 一元 XOR 位取反
		{
			name:    `compl`,
			evalstr: `^5`,
			args:    nil,
			want:    `-6`,
			wantErr: false,
		},
		{
			name:    `compl_zero`,
			evalstr: `^0`,
			args:    nil,
			want:    `-1`,
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			e, err := New(tt.evalstr, pool)
			if !tt.wantErr && (err != nil) {
				t.Errorf("expr New error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			for i := 0; i < 2; i++ {
				val, err := e.Eval(tt.args)
				if !tt.wantErr && (err != nil) {
					t.Errorf("expr Eval error = %v, wantErr %v", err, tt.wantErr)
					return
				}
				if !tt.wantErr && val.String() != tt.want {
					t.Errorf("expr Result got = %v, want %v", val.String(), tt.want)
				}
			}
		})
	}
}

func TestExprErrors(t *testing.T) {
	pool, _ := NewPool()
	cases := []struct{ expr string }{
		{"a + 1"},    // missing var a
		{"(1 + 2"},   // syntax error
		{"foobar()"}, // undefined func
	}
	for _, c := range cases {
		_, err := Eval(c.expr, nil, pool)
		if err == nil {
			t.Errorf("expected error for expr %q", c.expr)
		}
	}
}

const BenchmarkExpr = `((2 << 3) + (10 % 3)) * (5 - (x * 2)) + (3.0 / y) * (2.0 + 1.0) && ((z + "World") == "Hello World")`

func BenchmarkExprNoCache(b *testing.B) {
	pool, _ := NewPool(WithBuiltinList(builtin))
	e, err := New(BenchmarkExpr, pool)
	if err != nil {
		b.Errorf("expr New error = %v", err)
		return
	}
	for i := 0; i < b.N; i++ {
		_, err := e.Eval(map[string]interface{}{
			"x": 3.0,
			"y": 2.0,
			"z": "Hello ",
		})
		if err != nil {
			b.Errorf("expr Eval error = %v", err)
			return
		}
	}
}

func BenchmarkExprCache(b *testing.B) {
	pool, _ := NewPool(WithBuiltinList(builtin))
	e, err := New(BenchmarkExpr, pool, WithCacheValues(true))
	if err != nil {
		b.Errorf("expr New error = %v", err)
		return
	}
	for i := 0; i < b.N; i++ {
		_, err := e.Eval(map[string]interface{}{
			"x": 3.0,
			"y": 2.0,
			"z": "Hello ",
		})
		if err != nil {
			b.Errorf("expr Eval error = %v", err)
			return
		}
	}
}

func BenchmarkBuiltinCall(b *testing.B) {
	f := func(x, y float64, z string) bool {
		if (z + "World") == "Hello World" {
			return int(((2<<3)+(10%3))*(5-int(x*2))+int((3.0/y)*(2.0+1.0))) != 0
		}
		return false
	}
	for i := 0; i < b.N; i++ {
		if !f(3.0, 2.0, "Hello ") {
			b.Errorf("expr Eval error = %v", false)
			return
		}
	}
}

func BenchmarkDeepNested(b *testing.B) {
	depth := 100
	parts := make([]string, depth)
	for i := 0; i < depth; i++ {
		parts[i] = "1+"
	}
	expr := strings.Repeat("(", depth) + strings.Join(parts, "") + "1" + strings.Repeat(")", depth)
	pool, _ := NewPool()
	e, err := New(expr, pool, WithCacheValues(true))
	if err != nil {
		b.Errorf("expr New error = %v", err)
		return
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		e.Eval(nil)
	}
}

// BenchmarkDeepNestedNative 原生 Go 循环实现对比
var nativeRes int

func BenchmarkDeepNestedNative(b *testing.B) {
	depth := 100
	var res int
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res = 1
		// 模拟深度嵌套累加
		for j := 0; j < depth; j++ {
			res += 1
		}
	}
	nativeRes = res
}
