package expr

import (
	"fmt"
	"testing"
)

func TestExpr1(t *testing.T) {
	pool, _ := NewPool(map[string]BuiltinFunc{
		"sum": func(args ...interface{}) (interface{}, error) {
			var sum int64
			for _, v := range args {
				v1, ok := v.(int64)
				if !ok {
					return nil, fmt.Errorf("%v isn't int64", v)
				}
				sum += v1
			}
			return sum, nil
		},
	})

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
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			e, err := New(tt.evalstr, pool)
			if !tt.wantErr && (err != nil) {
				t.Errorf("expr New error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			val, err := e.Eval(tt.args)
			if !tt.wantErr && (err != nil) {
				t.Errorf("expr Eval error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			t.Logf("%v = %v", tt.evalstr, val)
			if !tt.wantErr && val.String() != tt.want {
				t.Errorf("expr Result got = %v, want %v", val.String(), tt.want)
			}
		})
	}
}

func BenchmarkExpr1(b *testing.B) {
	pool, _ := NewPool(nil)
	e, err := New(`((2 << 3) + (10 % 3)) * (5 - (x * 2)) + (3.0 / y) * (2.0 + 1.0) && ((z + "World") == "Hello World")`, pool)
	if err != nil {
		b.Errorf("expr New error = %v", err)
		return
	}
	for i := 0; i < b.N; i++ {
		val, err := e.Eval(map[string]interface{}{
			"x": 3.0,
			"y": 2.0,
			"z": "Hello ",
		})
		if val.String() != `true` {
			b.Errorf("expr Eval error = %v", err)
			return
		}
	}
}
