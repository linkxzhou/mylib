package expr

import (
	"fmt"
	"testing"
)

func TestExample(t *testing.T) {
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
			name:    `sum`,
			evalstr: `(sum(1,2,3) + x + y + x*y + x/y) || (1+0)`,
			args: map[string]interface{}{
				"x": 122.0,
				"y": 123.0,
			},
			want:    `true`,
			wantErr: false,
		},
		{
			name:    `add1`,
			evalstr: `1 + 2 - 3 + 4 - 5 + 6`,
			args:    nil,
			want:    `5`,
			wantErr: false,
		},
		{
			name:    `mul1`,
			evalstr: `6/6 + 6/0`,
			args:    nil,
			want:    `1`,
			wantErr: true,
		},
		{
			name:    `mul2`,
			evalstr: `6.0/6.0`,
			args:    nil,
			want:    `1`,
			wantErr: false,
		},
		{
			name:    `float1`,
			evalstr: `6.12 + 6.34`,
			args:    nil,
			want:    `12.46`,
			wantErr: false,
		},
		{
			name:    `string`,
			evalstr: `"6.12" + "0.1111"`,
			args:    nil,
			want:    `6.120.1111`,
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
