package sjson

import (
	"io"
	"strconv"
	"strings"
	"sync"
	"unicode/utf8"
)

// TokenType 表示词法标记的类型
type TokenType int

const (
	InvalidToken      TokenType = iota // 无效的标记
	EOFToken                           // 文件结束标记
	NumberToken                        // 数字标记，例如：123, 45.67
	StringToken                        // 字符串标记，例如："hello"
	NullToken                          // null值标记
	TrueToken                          // true布尔值标记
	FalseToken                         // false布尔值标记
	CommaToken                         // 逗号标记 ','
	ColonToken                         // 冒号标记 ':'
	LeftBraceToken                     // 左大括号标记 '{'
	RightBraceToken                    // 右大括号标记 '}'
	LeftBracketToken                   // 左方括号标记 '['
	RightBracketToken                  // 右方括号标记 ']'
)

// Token 表示一个词法标记
type Token struct {
	Type  TokenType
	Value string
	Pos   int
}

// Lexer 用于将JSON文本转换为标记流
type Lexer struct {
	input string
	pos   int
	start int
	width int
}

// 用于复用 strings.Builder
var stringBuilderPool = sync.Pool{
	New: func() interface{} {
		return &strings.Builder{}
	},
}

// NewLexer 创建一个新的词法分析器
func NewLexer(input string) *Lexer {
	return &Lexer{input: input}
}

// NewLexerFromReader 从io.Reader创建一个新的词法分析器
func NewLexerFromReader(r io.Reader) (*Lexer, error) {
	// 从对象池获取Builder
	sb := stringBuilderPool.Get().(*strings.Builder)
	sb.Reset() // 确保是干净的状态

	// 复制内容
	_, err := io.Copy(sb, r)
	if err != nil {
		// 遇到错误时归还Builder
		stringBuilderPool.Put(sb)
		return nil, err
	}

	// 获取字符串
	s := sb.String()

	// 归还Builder到对象池
	stringBuilderPool.Put(sb)

	// 创建词法分析器
	return NewLexer(s), nil
}

// next 返回下一个字符并前进
func (l *Lexer) next() rune {
	if l.pos >= len(l.input) {
		l.width = 0
		return -1
	}
	r, w := utf8.DecodeRuneInString(l.input[l.pos:])
	l.width = w
	l.pos += w
	return r
}

// ignore 忽略当前标记中已扫描的文本
func (l *Lexer) ignore() {
	l.start = l.pos
}

// NextToken 返回下一个标记
func (l *Lexer) NextToken() Token {
	l.start = l.pos

	inputLen := len(l.input)

	// 快速跳过空白字符（ASCII空白）
	for l.pos < inputLen {
		c := l.input[l.pos]
		if c == ' ' || c == '\n' || c == '\t' || c == '\r' {
			l.pos++
		} else {
			break
		}
	}
	l.start = l.pos // 更新标记起始位置

	// 检查EOF
	if l.pos >= len(l.input) {
		return Token{Type: EOFToken, Value: "", Pos: l.start}
	}

	// 基于当前字节快速确定标记类型
	c := l.input[l.pos]
	l.pos++ // 提前移动位置，大多数单字符标记只需要读一个字节

	// 处理单字符标记（最常见的情况）
	switch c {
	case '{':
		return Token{Type: LeftBraceToken, Value: "{", Pos: l.start}
	case '}':
		return Token{Type: RightBraceToken, Value: "}", Pos: l.start}
	case '[':
		return Token{Type: LeftBracketToken, Value: "[", Pos: l.start}
	case ']':
		return Token{Type: RightBracketToken, Value: "]", Pos: l.start}
	case ',':
		return Token{Type: CommaToken, Value: ",", Pos: l.start}
	case ':':
		return Token{Type: ColonToken, Value: ":", Pos: l.start}
	case '"': // 字符串
		l.pos-- // 回退，因为lexString需要读取引号
		return l.lexString()
	case '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9': // 数字
		l.pos-- // 回退
		return l.lexNumber()
	}

	// 处理关键字：直接检查开头并内联比较
	if c == 'n' && inputLen-l.start >= 4 &&
		l.input[l.start:l.start+4] == "null" {
		l.pos = l.start + 4
		return Token{Type: NullToken, Value: "null", Pos: l.start}
	} else if c == 't' && inputLen-l.start >= 4 &&
		l.input[l.start:l.start+4] == "true" {
		l.pos = l.start + 4
		return Token{Type: TrueToken, Value: "true", Pos: l.start}
	} else if c == 'f' && inputLen-l.start >= 5 &&
		l.input[l.start:l.start+5] == "false" {
		l.pos = l.start + 5
		return Token{Type: FalseToken, Value: "false", Pos: l.start}
	}

	// 无效标记
	return Token{Type: InvalidToken, Value: string(c), Pos: l.start}
}

// lexString 解析字符串标记
func (l *Lexer) lexString() Token {
	startPos := l.start // 保存标记开始位置

	// 跳过起始引号
	c := l.next()
	if c != '"' {
		// 这不应该发生，因为 NextToken 已经检查了第一个字符
		return Token{Type: InvalidToken, Value: string(c), Pos: startPos}
	}
	l.ignore() // 忽略起始引号，start 指向内容开始

	sb := stringBuilderPool.Get().(*strings.Builder)
	sb.Reset()
	// 使用 defer 确保 builder 被归还，即使发生 panic
	defer stringBuilderPool.Put(sb)

	// 优化：处理非转义字符块
	chunkStart := l.pos
	// 预分配内存减少重新分配
	sb.Grow(32) // 为常见字符串预分配一些空间

	// 快速路径：无转义字符的情况
	endQuotePos := strings.IndexByte(l.input[l.pos:], '"')
	if endQuotePos >= 0 && !strings.ContainsRune(l.input[l.pos:l.pos+endQuotePos], '\\') {
		// 没有转义字符，直接提取字符串
		value := l.input[l.pos : l.pos+endQuotePos]
		l.pos += endQuotePos + 1 // +1 跳过结束引号
		return Token{Type: StringToken, Value: value, Pos: startPos}
	}

	// 慢路径：处理带转义字符的情况
	for {
		c = l.next()
		if c == '\\' {
			// 追加反斜杠之前的块
			if l.pos > chunkStart+l.width { // 检查是否有内容需要追加 (l.width 是 '\' 的宽度)
				sb.WriteString(l.input[chunkStart : l.pos-l.width])
			}

			// 处理转义序列
			esc := l.next()
			if esc == -1 { // 反斜杠后遇到EOF
				return Token{Type: InvalidToken, Value: "未闭合的字符串 (EOF after escape)", Pos: startPos}
			}
			switch esc {
			case '"', '\\', '/':
				sb.WriteRune(esc)
			case 'b':
				sb.WriteRune('\b')
			case 'f':
				sb.WriteRune('\f')
			case 'n':
				sb.WriteRune('\n')
			case 'r':
				sb.WriteRune('\r')
			case 't':
				sb.WriteRune('\t')
			case 'u':
				// 检查是否有足够的字符用于 \uXXXX
				if l.pos+4 > len(l.input) {
					return Token{Type: InvalidToken, Value: "无效的 Unicode 转义序列 (过短)", Pos: l.pos - 1}
				}
				hex := l.input[l.pos : l.pos+4]
				code, err := strconv.ParseInt(hex, 16, 32)
				if err != nil {
					return Token{Type: InvalidToken, Value: "无效的 Unicode 转义序列: " + hex, Pos: l.pos - 1}
				}
				l.pos += 4 // 跳过4个十六进制数字

				// 处理Unicode代理对 (surrogate pairs)
				if code >= 0xD800 && code <= 0xDBFF { // 高代理项 (high surrogate)
					// 检查是否紧跟着另一个 \u 转义序列
					if l.pos+6 <= len(l.input) && l.input[l.pos] == '\\' && l.input[l.pos+1] == 'u' {
						// 提前移动位置到下一个 \u 后面
						l.pos += 2

						// 读取低代理项 (low surrogate)
						if l.pos+4 > len(l.input) {
							return Token{Type: InvalidToken, Value: "无效的 Unicode 代理对 (过短)", Pos: l.pos - 1}
						}

						lowHex := l.input[l.pos : l.pos+4]
						lowCode, err := strconv.ParseInt(lowHex, 16, 32)
						if err != nil {
							return Token{Type: InvalidToken, Value: "无效的 Unicode 代理对: " + lowHex, Pos: l.pos - 1}
						}

						// 检查是否是有效的低代理项
						if lowCode >= 0xDC00 && lowCode <= 0xDFFF {
							// 计算完整的Unicode代码点
							// 公式: (highSurrogate - 0xD800) * 0x400 + (lowSurrogate - 0xDC00) + 0x10000
							fullCode := (code-0xD800)*0x400 + (lowCode - 0xDC00) + 0x10000
							sb.WriteRune(rune(fullCode))
							l.pos += 4 // 跳过低代理项的4个十六进制数字
						} else {
							// 不是有效的低代理项，回退位置并只处理高代理项
							l.pos -= 2 // 回退到 \u 之前
							sb.WriteRune(rune(code))
						}
					} else {
						// 没有紧跟着的低代理项，只处理单个 \u 转义序列
						sb.WriteRune(rune(code))
					}
				} else {
					// 普通的 Unicode 字符
					sb.WriteRune(rune(code))
				}
			default:
				// 无效的转义序列
				// 定位到反斜杠的位置 (当前 pos - 转义符宽度 - 反斜杠宽度)
				return Token{Type: InvalidToken, Value: "无效的转义字符: \\" + string(esc), Pos: l.pos - l.width*2}
			}
			// 更新下一个块的起始位置
			chunkStart = l.pos

		} else if c == '"' {
			// 追加结束引号之前的最后一个块
			if l.pos > chunkStart+l.width { // 检查是否有内容需要追加 (l.width 是 '"' 的宽度)
				sb.WriteString(l.input[chunkStart : l.pos-l.width])
			}
			// 字符串结束
			break
		} else if c == -1 { // 在结束引号之前遇到EOF
			return Token{Type: InvalidToken, Value: "未闭合的字符串", Pos: startPos}
		}
		// 如果是普通字符，则继续循环，它是当前块的一部分
	}

	// 注意：Token 的 Pos 应该是原始字符串中标记的起始位置，包括引号
	return Token{Type: StringToken, Value: sb.String(), Pos: startPos}
}

// lexNumber 解析数字标记
func (l *Lexer) lexNumber() Token {
	startPos := l.start
	inputLen := len(l.input)
	// 直接扫描数字，避免多次函数调用
	// 1. 处理负号
	if l.pos < inputLen && l.input[l.pos] == '-' {
		l.pos++
	}

	// 2. 处理整数部分
	if l.pos < inputLen && l.input[l.pos] == '0' {
		l.pos++
		// 如果以0开头，后面不能直接跟数字 (除非是 0.xxx)
		if l.pos < inputLen && l.input[l.pos] >= '0' && l.input[l.pos] <= '9' {
			return Token{Type: InvalidToken, Value: "无效的数字格式 (以0开头)", Pos: startPos}
		}
	} else if l.pos < inputLen && l.input[l.pos] >= '1' && l.input[l.pos] <= '9' {
		l.pos++ // 第一个数字
		// 扫描剩余数字
		for l.pos < inputLen && l.input[l.pos] >= '0' && l.input[l.pos] <= '9' {
			l.pos++
		}
	} else {
		// 没有整数部分
		return Token{Type: InvalidToken, Value: "无效的数字格式 (缺少整数部分)", Pos: startPos}
	}

	// 3. 处理小数部分
	if l.pos < inputLen && l.input[l.pos] == '.' {
		l.pos++ // 跳过小数点

		// 小数点后必须有数字
		if l.pos >= inputLen || l.input[l.pos] < '0' || l.input[l.pos] > '9' {
			return Token{Type: InvalidToken, Value: "无效的数字格式 (小数点后无数字)", Pos: l.pos - 1}
		}

		// 扫描小数部分的数字
		for l.pos < inputLen {
			c := l.input[l.pos]
			if c < '0' || c > '9' {
				break
			}
			l.pos++
		}
	}

	// 4. 处理指数部分
	if l.pos < inputLen && (l.input[l.pos] == 'e' || l.input[l.pos] == 'E') {
		l.pos++ // 跳过 e/E

		// 处理可选的 +/-
		if l.pos < inputLen && (l.input[l.pos] == '+' || l.input[l.pos] == '-') {
			l.pos++
		}

		// e/E 后面必须有数字
		if l.pos >= inputLen || l.input[l.pos] < '0' || l.input[l.pos] > '9' {
			return Token{Type: InvalidToken, Value: "无效的数字格式 (指数后无数字)", Pos: l.pos - 1}
		}

		// 扫描指数部分的数字
		for l.pos < inputLen {
			c := l.input[l.pos]
			if c < '0' || c > '9' {
				break
			}
			l.pos++
		}
	}

	numStr := l.input[startPos:l.pos]
	return Token{Type: NumberToken, Value: numStr, Pos: startPos}
}

// ParseAll 返回所有标记，直到遇到 EOF
func (l *Lexer) ParseAll() []Token {
	var tokens []Token
	for {
		token := l.NextToken()
		tokens = append(tokens, token)
		if token.Type == EOFToken {
			break
		}
	}
	return tokens
}
