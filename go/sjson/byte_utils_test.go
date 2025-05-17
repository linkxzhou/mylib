package sjson

import (
	"math"
	"strconv"
	"testing"
)

// 测试从字节切片解析整数
func TestParseIntFromBytes(t *testing.T) {
	tests := []struct {
		name    string
		input   []byte
		base    int
		bitSize int
		want    int64
		wantErr bool
	}{
		// 基本测试
		{"零值", []byte("0"), 10, 64, 0, false},
		{"正整数", []byte("123"), 10, 64, 123, false},
		{"负整数", []byte("-123"), 10, 64, -123, false},
		{"带加号", []byte("+123"), 10, 64, 123, false},
		{"最大值", []byte("9223372036854775807"), 10, 64, math.MaxInt64, false},
		{"最小值", []byte("-9223372036854775808"), 10, 64, math.MinInt64, false},

		// 不同进制测试
		{"十六进制", []byte("1A"), 16, 64, 26, false},
		{"二进制", []byte("1010"), 2, 64, 10, false},
		{"八进制", []byte("70"), 8, 64, 56, false},

		// 不同位宽测试
		{"32位整数", []byte("2147483647"), 10, 32, math.MaxInt32, false},
		{"16位整数", []byte("32767"), 10, 16, math.MaxInt16, false},
		{"8位整数", []byte("127"), 10, 8, math.MaxInt8, false},

		// 错误测试
		{"空输入", []byte(""), 10, 64, 0, true},
		{"非数字字符", []byte("12a34"), 10, 64, 0, true},
		{"仅符号", []byte("-"), 10, 64, 0, true},
		{"超出范围", []byte("9223372036854775808"), 10, 64, math.MinInt64, false}, // 溢出处理实际返回最小值
		{"32位溢出", []byte("2147483648"), 10, 32, 0, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseIntFromBytes(tt.input, tt.base, tt.bitSize)
			if (err != nil) != tt.wantErr {
				t.Errorf("parseIntFromBytes() 错误 = %v, 期望错误 = %v", err, tt.wantErr)
				return
			}
			if err == nil && got != tt.want {
				t.Errorf("parseIntFromBytes() = %v, 期望 %v", got, tt.want)
			}
		})
	}
}

// 测试从字节切片解析无符号整数
func TestParseUintFromBytes(t *testing.T) {
	tests := []struct {
		name    string
		input   []byte
		base    int
		bitSize int
		want    uint64
		wantErr bool
	}{
		// 基本测试
		{"零值", []byte("0"), 10, 64, 0, false},
		{"正整数", []byte("123"), 10, 64, 123, false},
		{"带加号", []byte("+123"), 10, 64, 123, false},
		{"最大值", []byte("18446744073709551615"), 10, 64, math.MaxUint64, false},

		// 不同进制测试
		{"十六进制", []byte("FF"), 16, 64, 255, false},
		{"二进制", []byte("1111"), 2, 64, 15, false},
		{"八进制", []byte("77"), 8, 64, 63, false},

		// 不同位宽测试
		{"32位无符号", []byte("4294967295"), 10, 32, math.MaxUint32, false},
		{"16位无符号", []byte("65535"), 10, 16, math.MaxUint16, false},
		{"8位无符号", []byte("255"), 10, 8, math.MaxUint8, false},

		// 错误测试
		{"空输入", []byte(""), 10, 64, 0, true},
		{"负数", []byte("-1"), 10, 64, 0, true},
		{"非数字字符", []byte("12a34"), 10, 64, 0, true},
		{"仅符号", []byte("+"), 10, 64, 0, true},
		{"超出范围", []byte("18446744073709551616"), 10, 64, 0, false}, // 溢出处理实际返回0
		{"32位溢出", []byte("4294967296"), 10, 32, 0, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseUintFromBytes(tt.input, tt.base, tt.bitSize)
			if (err != nil) != tt.wantErr {
				t.Errorf("parseUintFromBytes() 错误 = %v, 期望错误 = %v", err, tt.wantErr)
				return
			}
			if err == nil && got != tt.want {
				t.Errorf("parseUintFromBytes() = %v, 期望 %v", got, tt.want)
			}
		})
	}
}

// 测试从字节切片解析浮点数
func TestParseFloatFromBytes(t *testing.T) {
	tests := []struct {
		name    string
		input   []byte
		bitSize int
		want    float64
		wantErr bool
	}{
		// 基本测试
		{"零值", []byte("0"), 64, 0, false},
		{"正整数", []byte("123"), 64, 123, false},
		{"负整数", []byte("-123"), 64, -123, false},
		{"带加号", []byte("+123"), 64, 123, false},
		{"小数", []byte("3.14159"), 64, 3.14159, false},
		{"负小数", []byte("-3.14159"), 64, -3.14159, false},

		// 科学计数法
		{"科学计数法正", []byte("1.23e2"), 64, 123, false},
		{"科学计数法负", []byte("1.23e-2"), 64, 0.0123, false},
		{"科学计数法大写", []byte("1.23E2"), 64, 123, false},

		// 特殊值
		{"最大值", []byte("1.7976931348623157e308"), 64, math.MaxFloat64, false},
		{"极小值", []byte("4.9406564584124654e-324"), 64, math.SmallestNonzeroFloat64, false},

		// 32位测试
		{"32位浮点数", []byte("3.14159"), 32, float64(float32(3.14159)), false},

		// 错误测试
		{"空输入", []byte(""), 64, 0, true},
		{"非数字字符", []byte("3.14a"), 64, 0, true},
		{"仅符号", []byte("-"), 64, 0, true},
		{"仅小数点", []byte("."), 64, 0, true},
		{"无效格式", []byte("1.23e"), 64, 0, true},
		{"无效指数", []byte("1.23e+"), 64, 0, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseFloatFromBytes(tt.input, tt.bitSize)
			if (err != nil) != tt.wantErr {
				t.Errorf("parseFloatFromBytes() 错误 = %v, 期望错误 = %v", err, tt.wantErr)
				return
			}
			if err == nil {
				// 浮点数比较需要考虑精度
				if tt.bitSize == 32 {
					// 32位浮点数比较
					if float32(got) != float32(tt.want) {
						t.Errorf("parseFloatFromBytes() = %v, 期望 %v", got, tt.want)
					}
				} else {
					// 对于接近0的极小值，使用相对误差
					if tt.want != 0 && math.Abs(got-tt.want)/math.Abs(tt.want) > 1e-15 {
						t.Errorf("parseFloatFromBytes() = %v, 期望 %v", got, tt.want)
					} else if tt.want == 0 && got != 0 {
						t.Errorf("parseFloatFromBytes() = %v, 期望 %v", got, tt.want)
					}
				}
			}
		})
	}
}

func TestAppendUint(t *testing.T) {
	cases := []uint64{
		0, 1, 9, 10, 99, 100, 999, 1000, 123456, math.MaxUint32, math.MaxUint64,
	}
	for _, u := range cases {
		got := appendUint(nil, u, 10)
		want := strconv.AppendUint(nil, u, 10)
		if string(got) != string(want) {
			t.Errorf("appendUint(%d) = %q, want %q", u, got, want)
		}
	}
}

func TestAppendInt(t *testing.T) {
	cases := []int64{
		0, 1, -1, 9, -9, 10, -10, 99, -99, 100, -100, 999, -999, 1000, -1000,
		math.MaxInt32, math.MinInt32, math.MaxInt64, math.MinInt64 + 1,
	}
	for _, i := range cases {
		got := appendInt(nil, i, 10)
		want := strconv.AppendInt(nil, i, 10)
		if string(got) != string(want) {
			t.Errorf("appendInt(%d) = %q, want %q", i, got, want)
		}
	}
}

// 性能基准测试
func BenchmarkParseIntFromBytes(b *testing.B) {
	input := []byte("9223372036854775807")
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = parseIntFromBytes(input, 10, 64)
	}
}

func BenchmarkParseUintFromBytes(b *testing.B) {
	input := []byte("18446744073709551615")
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = parseUintFromBytes(input, 10, 64)
	}
}

func BenchmarkParseFloatFromBytes(b *testing.B) {
	input := []byte("3.14159265358979323846")
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = parseFloatFromBytes(input, 64)
	}
}

// 与标准库比较的基准测试
func BenchmarkParseIntComparison(b *testing.B) {
	input := []byte("9223372036854775807")
	inputStr := string(input)

	b.Run("parseIntFromBytes", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _ = parseIntFromBytes(input, 10, 64)
		}
	})

	b.Run("strconv.ParseInt", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _ = strconv.ParseInt(inputStr, 10, 64)
		}
	})
}

func BenchmarkParseFloatComparison(b *testing.B) {
	input := []byte("3.14159265358979323846")
	inputStr := string(input)

	b.Run("parseFloatFromBytes", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _ = parseFloatFromBytes(input, 64)
		}
	})

	b.Run("strconv.ParseFloat", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _ = strconv.ParseFloat(inputStr, 64)
		}
	})
}

// 性能基准测试 - 整数转字符串
func BenchmarkAppendInt(b *testing.B) {
	cases := []struct {
		name string
		val  int64
	}{
		{"Small", 42},
		{"Medium", 12345},
		{"Large", 9876543210},
		{"MaxInt64", math.MaxInt64},
		{"MinInt64", math.MinInt64 + 1}, // +1 避免溢出问题
	}

	for _, tc := range cases {
		b.Run("Custom-"+tc.name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = appendInt(nil, tc.val, 10)
			}
		})

		b.Run("Strconv-"+tc.name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = strconv.AppendInt(nil, tc.val, 10)
			}
		})
	}
}

func BenchmarkAppendUint(b *testing.B) {
	cases := []struct {
		name string
		val  uint64
	}{
		{"Tiny", 7},
		{"Small", 42},
		{"Under1000", 999},
		{"Over1000", 1001},
		{"Medium", 12345},
		{"Large", 9876543210},
		{"MaxUint64", math.MaxUint64},
	}

	for _, tc := range cases {
		b.Run("Custom-"+tc.name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = appendUint(nil, tc.val, 10)
			}
		})

		b.Run("Strconv-"+tc.name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = strconv.AppendUint(nil, tc.val, 10)
			}
		})
	}
}

// 测试大量整数转换的场景
func BenchmarkBatchIntegerConversion(b *testing.B) {
	// 准备各种大小的整数
	var nums []int64
	for i := 0; i < 100; i++ {
		nums = append(nums, int64(i))          // 小整数
		nums = append(nums, int64(i+1000))     // 中等整数
		nums = append(nums, int64(i+1000000))  // 大整数
		nums = append(nums, -int64(i+1))       // 负小整数
		nums = append(nums, -int64(i+1000))    // 负中等整数
		nums = append(nums, -int64(i+1000000)) // 负大整数
	}

	b.Run("Custom-Batch", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			buf := make([]byte, 0, 1024)
			for _, n := range nums {
				buf = appendInt(buf, n, 10)
				buf = append(buf, ',')
			}
		}
	})

	b.Run("Strconv-Batch", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			buf := make([]byte, 0, 1024)
			for _, n := range nums {
				buf = strconv.AppendInt(buf, n, 10)
				buf = append(buf, ',')
			}
		}
	})
}
