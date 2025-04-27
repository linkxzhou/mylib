package simplejs

import (
	"bufio"
	"fmt"
	"io"
	"strings"
)

// TokenType defines the type of lexical tokens.
type TokenType int

const (
	TokEOF            TokenType = iota // EOF
	TokNumber                          // 123, 3.14
	TokString                          // "hello"
	TokIdentifier                      // foo, bar
	TokBool                            // true, false
	TokNull                            // null
	TokUndefined                       // undefined
	TokLParen                          // (
	TokRParen                          // )
	TokLBrace                          // {
	TokRBrace                          // }
	TokLBracket                        // [
	TokRBracket                        // ]
	TokSemicolon                       // ;
	TokQuestion                        // ?
	TokColon                           // :
	TokAssign                          // =
	TokEqual                           // ==
	TokNotEqual                        // !=
	TokStrictEqual                     // ===
	TokNotStrictEqual                  // !==
	TokMinusAssign                     // -=
	TokAsteriskAssign                  // *=
	TokSlashAssign                     // /=
	TokComma                           // ,
	TokPlus                            // +
	TokMinus                           // -
	TokAsterisk                        // *
	TokSlash                           // /
	TokRem                             // %
	TokLT                              // <
	TokLTE                             // <=
	TokGT                              // >
	TokGTE                             // >=
	TokLShift                          // <<
	TokRShift                          // >>
	TokURShift                         // >>>
	TokLogicalAnd                      // &&
	TokLogicalOr                       // ||
	TokLogicalNot                      // !
	TokBitAnd                          // &
	TokBitOr                           // |
	TokBitXor                          // ^
	TokBitNot                          // ~
	TokDot                             // .
	TokDelete                          // delete keyword
	TokInc                             // ++
	TokDec                             // --
	TokPlusAssign                      // +=
	TokTemplate                        // template string literal
	TokSpread                          // ... spread/rest
	TokRegex                           // regex literal

	// keywords
	TokIf       // if
	TokElse     // else
	TokWhile    // while
	TokFor      // for
	TokFunction // function
	TokReturn   // return
	TokVar      // var
	TokLet      // let
	TokConst    // const
	TokArrow    // =>
	TokClass    // class
	TokExtends  // extends
	TokNew      // new
	TokSuper    // super
	TokThrow    // throw
	TokTry      // try
	TokCatch    // catch
	TokBreak    // break
)

// Token represents a lexical token.
type Token struct {
	Type    TokenType
	Literal string
	Line    int
}

func (t TokenType) String() string {
	switch t {
	case TokEOF:
		return "EOF"
	case TokNumber:
		return "Number"
	case TokString:
		return "String"
	case TokIdentifier:
		return "Identifier"
	case TokBool:
		return "Bool"
	case TokNull:
		return "Null"
	case TokUndefined:
		return "Undefined"
	case TokLParen:
		return "("
	case TokRParen:
		return ")"
	case TokLBrace:
		return "{"
	case TokRBrace:
		return "}"
	case TokLBracket:
		return "["
	case TokRBracket:
		return "]"
	case TokSemicolon:
		return ";"
	case TokQuestion:
		return "?"
	case TokColon:
		return ":"
	case TokAssign:
		return "="
	case TokEqual:
		return "=="
	case TokNotEqual:
		return "!="
	case TokStrictEqual:
		return "==="
	case TokNotStrictEqual:
		return "!=="
	case TokMinusAssign:
		return "-="
	case TokAsteriskAssign:
		return "*="
	case TokSlashAssign:
		return "/="
	case TokComma:
		return ","
	case TokPlus:
		return "+"
	case TokMinus:
		return "-"
	case TokAsterisk:
		return "*"
	case TokSlash:
		return "/"
	case TokRem:
		return "%"
	case TokLT:
		return "<"
	case TokLTE:
		return "<="
	case TokGT:
		return ">"
	case TokGTE:
		return ">="
	case TokLShift:
		return "<<"
	case TokRShift:
		return ">>"
	case TokURShift:
		return ">>>"
	case TokLogicalAnd:
		return "&&"
	case TokLogicalOr:
		return "||"
	case TokLogicalNot:
		return "!"
	case TokBitAnd:
		return "&"
	case TokBitOr:
		return "|"
	case TokBitXor:
		return "^"
	case TokBitNot:
		return "~"
	case TokDot:
		return "."
	case TokDelete:
		return "delete"
	case TokInc:
		return "++"
	case TokDec:
		return "--"
	case TokPlusAssign:
		return "+="
	case TokTemplate:
		return "template string literal"
	case TokSpread:
		return "..."
	case TokRegex:
		return "regex literal"
	case TokIf:
		return "if"
	case TokElse:
		return "else"
	case TokWhile:
		return "while"
	case TokFor:
		return "for"
	case TokFunction:
		return "function"
	case TokReturn:
		return "return"
	case TokVar:
		return "var"
	case TokLet:
		return "let"
	case TokConst:
		return "const"
	case TokArrow:
		return "=>"
	case TokClass:
		return "class"
	case TokExtends:
		return "extends"
	case TokNew:
		return "new"
	case TokSuper:
		return "super"
	case TokThrow:
		return "throw"
	case TokTry:
		return "try"
	case TokCatch:
		return "catch"
	case TokBreak:
		return "break"
	default:
		return fmt.Sprintf("TokenType(%d)", t)
	}
}

// scanner 按字节读取并可回退
type scanner struct {
	r    *bufio.Reader
	line int
}

func newScanner(input string) *scanner {
	return &scanner{r: bufio.NewReader(strings.NewReader(input)), line: 1}
}

// next 读取下一个字节，遇到换行则行号 +1
func (s *scanner) next() (byte, error) {
	ch, err := s.r.ReadByte()
	if err != nil {
		return 0, err
	}
	if ch == '\n' {
		s.line++
	}
	return ch, nil
}

// peek 预读一个字节，不移动游标
func (s *scanner) peek() (byte, error) {
	bs, err := s.r.Peek(1)
	if err != nil {
		return 0, err
	}
	return bs[0], nil
}

// unread 将最近一次 ReadByte 回退
func (s *scanner) unread() error {
	return s.r.UnreadByte()
}

// Tokenize 使用 scanner 实现输入扫描，支持注释、标识符、数字、字符串、正则等
func Tokenize(input string) ([]Token, error) {
	s := newScanner(input)
	var tokens []Token
	for {
		ch, err := s.next()
		if err != nil {
			if err == io.EOF {
				break
			}
			return nil, err
		}
		// 跳过空白
		if ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n' {
			continue
		}
		// 此处按原逻辑分支：注释(/, /*), 标识符/关键字, 数字, 字符串, 正则, 运算符等
		// 使用 s.peek()/s.unread() 处理多字符符号
		// 具体实现请参考原 Tokenize 内部逻辑，将索引换成 next/peek/unread
		// 示例：单字符运算符
		switch ch {
		case '+':
			b, err := s.peek()
			if err == nil {
				if b == '=' {
					s.next()
					tokens = append(tokens, Token{Type: TokPlusAssign, Literal: "+=", Line: s.line})
					continue
				} else if b == '+' {
					s.next()
					tokens = append(tokens, Token{Type: TokInc, Literal: "++", Line: s.line})
					continue
				}
			}
			tokens = append(tokens, Token{Type: TokPlus, Literal: "+", Line: s.line})
			continue
		case '-':
			b, err := s.peek()
			if err == nil {
				if b == '=' {
					s.next()
					tokens = append(tokens, Token{Type: TokMinusAssign, Literal: "-=", Line: s.line})
					continue
				} else if b == '-' {
					s.next()
					tokens = append(tokens, Token{Type: TokDec, Literal: "--", Line: s.line})
					continue
				}
			}
			tokens = append(tokens, Token{Type: TokMinus, Literal: "-", Line: s.line})
			continue
		case '*':
			b, err := s.peek()
			if err == nil && b == '=' {
				s.next()
				tokens = append(tokens, Token{Type: TokAsteriskAssign, Literal: "*=", Line: s.line})
				continue
			}
			tokens = append(tokens, Token{Type: TokAsterisk, Literal: "*", Line: s.line})
			continue
		case '/':
			b, err := s.peek()
			if err == nil && b == '=' {
				s.next()
				tokens = append(tokens, Token{Type: TokSlashAssign, Literal: "/=", Line: s.line})
				continue
			}
			// 注释处理
			b, err = s.peek()
			if err != nil && err != io.EOF {
				return nil, err
			}
			if err == nil && b == '/' {
				// 单行注释
				s.next() // consume second '/'
				for {
					ch, err = s.next()
					if err != nil || ch == '\n' {
						if ch == '\n' {
							s.line++
						}
						break
					}
				}
				continue
			}
			if err == nil && b == '*' {
				// 多行注释
				s.next() // consume '*'
				for {
					ch, err = s.next()
					if err != nil {
						break
					}
					if ch == '*' {
						c, err2 := s.peek()
						if err2 != nil {
							break
						}
						if c == '/' {
							s.next()
							break
						}
					}
					if ch == '\n' {
						s.line++
					}
				}
				continue
			}
			// regex literal
			if isRegexContext(tokens) {
				start := s.line
				lit := string(ch) // '/'
				for {
					ch, err = s.next()
					if err != nil {
						return nil, err
					}
					lit += string(ch)
					if ch == '/' {
						break
					}
					if ch == '\n' {
						s.line++
					}
				}
				tokens = append(tokens, Token{Type: TokRegex, Literal: lit, Line: start})
				continue
			}
			// 普通 '/'
			tokens = append(tokens, Token{Type: TokSlash, Literal: "/", Line: s.line})
			continue
		case '=':
			// 优先 ===
			b, err := s.peek()
			if err != nil && err != io.EOF {
				return nil, err
			}
			if b == '=' {
				s.next()
				b2, err := s.peek()
				if err != nil && err != io.EOF {
					return nil, err
				}
				if b2 == '=' {
					s.next()
					tokens = append(tokens, Token{Type: TokStrictEqual, Literal: "===", Line: s.line})
					continue
				}
				tokens = append(tokens, Token{Type: TokEqual, Literal: "==", Line: s.line})
				continue
			}
			b, err = s.peek()
			if err != nil && err != io.EOF {
				return nil, err
			}
			if b == '>' {
				s.next()
				tokens = append(tokens, Token{Type: TokArrow, Literal: "=>", Line: s.line})
				continue
			}
			tokens = append(tokens, Token{Type: TokAssign, Literal: "=", Line: s.line})
			continue
		case '!':
			b, err := s.peek()
			if err != nil && err != io.EOF {
				return nil, err
			}
			if b == '=' {
				s.next()
				b2, err := s.peek()
				if err != nil && err != io.EOF {
					return nil, err
				}
				if b2 == '=' {
					s.next()
					tokens = append(tokens, Token{Type: TokNotStrictEqual, Literal: "!==", Line: s.line})
					continue
				}
				tokens = append(tokens, Token{Type: TokNotEqual, Literal: "!=", Line: s.line})
				continue
			}
			tokens = append(tokens, Token{Type: TokLogicalNot, Literal: "!", Line: s.line})
			continue
		case '%':
			tokens = append(tokens, Token{Type: TokRem, Literal: "%", Line: s.line})
		case '<':
			b, err := s.peek()
			if err == nil {
				if b == '<' {
					s.next()
					b2, err := s.peek()
					if err == nil && b2 == '=' {
						s.next()
						// 可选：如果实现 <<=
						tokens = append(tokens, Token{Type: TokLShift, Literal: "<<", Line: s.line})
						continue
					}
					tokens = append(tokens, Token{Type: TokLShift, Literal: "<<", Line: s.line})
					continue
				} else if b == '=' {
					s.next()
					tokens = append(tokens, Token{Type: TokLTE, Literal: "<=", Line: s.line})
					continue
				}
			}
			tokens = append(tokens, Token{Type: TokLT, Literal: "<", Line: s.line})
			continue
		case '>':
			// 识别 >>> 或 >>
			b, err := s.peek()
			if err == nil {
				if b == '>' {
					s.next()
					b2, err := s.peek()
					if err == nil && b2 == '>' {
						s.next()
						tokens = append(tokens, Token{Type: TokURShift, Literal: ">>>", Line: s.line})
						continue
					}
					tokens = append(tokens, Token{Type: TokRShift, Literal: ">>", Line: s.line})
					continue
				} else if b == '=' {
					s.next()
					tokens = append(tokens, Token{Type: TokGTE, Literal: ">=", Line: s.line})
					continue
				}
			}
			// 单个 >
			tokens = append(tokens, Token{Type: TokGT, Literal: ">", Line: s.line})
		case '&':
			b, err := s.peek()
			if err == nil && b == '&' {
				s.next()
				tokens = append(tokens, Token{Type: TokLogicalAnd, Literal: "&&", Line: s.line})
				continue
			}
			tokens = append(tokens, Token{Type: TokBitAnd, Literal: "&", Line: s.line})
			continue
		case '|':
			b, err := s.peek()
			if err == nil && b == '|' {
				s.next()
				tokens = append(tokens, Token{Type: TokLogicalOr, Literal: "||", Line: s.line})
				continue
			}
			tokens = append(tokens, Token{Type: TokBitOr, Literal: "|", Line: s.line})
			continue
		case '^':
			tokens = append(tokens, Token{Type: TokBitXor, Literal: "^", Line: s.line})
			continue
		case '~':
			tokens = append(tokens, Token{Type: TokBitNot, Literal: "~", Line: s.line})
			continue
		case '.':
			b, err := s.peek()
			if err == nil && b == '.' {
				s.next()
				b2, err := s.peek()
				if err == nil && b2 == '.' {
					s.next()
					tokens = append(tokens, Token{Type: TokSpread, Literal: "...", Line: s.line})
					continue
				}
			}
			// 单字符 '.'
			tokens = append(tokens, Token{Type: TokDot, Literal: ".", Line: s.line})
			continue
		case '(':
			tokens = append(tokens, Token{Type: TokLParen, Literal: "(", Line: s.line})
			continue
		case ')':
			tokens = append(tokens, Token{Type: TokRParen, Literal: ")", Line: s.line})
			continue
		case '{':
			tokens = append(tokens, Token{Type: TokLBrace, Literal: "{", Line: s.line})
			continue
		case '}':
			tokens = append(tokens, Token{Type: TokRBrace, Literal: "}", Line: s.line})
			continue
		case '[':
			tokens = append(tokens, Token{Type: TokLBracket, Literal: "[", Line: s.line})
			continue
		case ']':
			tokens = append(tokens, Token{Type: TokRBracket, Literal: "]", Line: s.line})
			continue
		case ',':
			tokens = append(tokens, Token{Type: TokComma, Literal: ",", Line: s.line})
			continue
		case ';':
			tokens = append(tokens, Token{Type: TokSemicolon, Literal: ";", Line: s.line})
			continue
		case ':':
			tokens = append(tokens, Token{Type: TokColon, Literal: ":", Line: s.line})
			continue
		case '?':
			tokens = append(tokens, Token{Type: TokQuestion, Literal: "?", Line: s.line})
			continue
		default:
			// identifier or keyword
			if isIdentBegin(ch) {
				start := s.line
				literal := string(ch)
				for {
					ch, err = s.next()
					if err != nil {
						break
					}
					if !isIdentContinue(ch) {
						_ = s.unread()
						break
					}
					literal += string(ch)
				}
				switch literal {
				case "true", "false":
					tokens = append(tokens, Token{Type: TokBool, Literal: literal, Line: start})
				case "null":
					tokens = append(tokens, Token{Type: TokNull, Literal: literal, Line: start})
				case "undefined":
					tokens = append(tokens, Token{Type: TokUndefined, Literal: literal, Line: start})
				case "if":
					tokens = append(tokens, Token{Type: TokIf, Literal: literal, Line: start})
				case "else":
					tokens = append(tokens, Token{Type: TokElse, Literal: literal, Line: start})
				case "while":
					tokens = append(tokens, Token{Type: TokWhile, Literal: literal, Line: start})
				case "for":
					tokens = append(tokens, Token{Type: TokFor, Literal: literal, Line: start})
				case "function":
					tokens = append(tokens, Token{Type: TokFunction, Literal: literal, Line: start})
				case "return":
					tokens = append(tokens, Token{Type: TokReturn, Literal: literal, Line: start})
				case "var":
					tokens = append(tokens, Token{Type: TokVar, Literal: literal, Line: start})
				case "let":
					tokens = append(tokens, Token{Type: TokLet, Literal: literal, Line: start})
				case "const":
					tokens = append(tokens, Token{Type: TokConst, Literal: literal, Line: start})
				case "class":
					tokens = append(tokens, Token{Type: TokClass, Literal: literal, Line: start})
				case "extends":
					tokens = append(tokens, Token{Type: TokExtends, Literal: literal, Line: start})
				case "new":
					tokens = append(tokens, Token{Type: TokNew, Literal: literal, Line: start})
				case "super":
					tokens = append(tokens, Token{Type: TokSuper, Literal: literal, Line: start})
				case "throw":
					tokens = append(tokens, Token{Type: TokThrow, Literal: literal, Line: start})
				case "try":
					tokens = append(tokens, Token{Type: TokTry, Literal: literal, Line: start})
				case "catch":
					tokens = append(tokens, Token{Type: TokCatch, Literal: literal, Line: start})
				case "break":
					tokens = append(tokens, Token{Type: TokBreak, Literal: literal, Line: start})
				case "delete":
					tokens = append(tokens, Token{Type: TokDelete, Literal: literal, Line: start})
				default:
					tokens = append(tokens, Token{Type: TokIdentifier, Literal: literal, Line: start})
				}
				continue
			}
			// number literal
			nextCh, errP := s.peek()
			if isDigit(ch) || (ch == '.' && errP == nil && nextCh != '.') {
				start := s.line
				literal := string(ch)
				for {
					ch, err = s.next()
					if err != nil {
						break
					}
					if !isDigit(ch) && ch != '.' {
						_ = s.unread()
						break
					}
					literal += string(ch)
				}
				tokens = append(tokens, Token{Type: TokNumber, Literal: literal, Line: start})
				continue
			}
			// string literal
			if ch == '"' || ch == '\'' {
				quote := ch
				start := s.line
				literal := ""
				for {
					ch, err = s.next()
					if err != nil {
						if err == io.EOF {
							return nil, fmt.Errorf("unterminated string")
						}
						return nil, err
					}
					if ch == '\\' {
						ch, err = s.next()
						if err != nil {
							return nil, err
						}
						literal += string(ch)
					} else if ch == quote {
						break
					} else {
						literal += string(ch)
					}
				}
				tokens = append(tokens, Token{Type: TokString, Literal: literal, Line: start})
				continue
			}
			// template string literal
			if ch == '`' {
				start := s.line
				literal := ""
				for {
					ch, err = s.next()
					if err != nil {
						return nil, err
					}
					if ch == '\\' {
						ch, err = s.next()
						if err != nil {
							return nil, err
						}
						literal += string(ch)
					} else if ch == '`' {
						break
					} else {
						literal += string(ch)
					}
					if ch == '\n' {
						s.line++
					}
				}
				tokens = append(tokens, Token{Type: TokTemplate, Literal: literal, Line: start})
				continue
			}
			return nil, fmt.Errorf("unexpected character: %c", ch)
		}
	}
	tokens = append(tokens, Token{Type: TokEOF, Literal: "", Line: s.line})
	return tokens, nil
}

func isDigit(ch byte) bool         { return ch >= '0' && ch <= '9' }
func isAlpha(ch byte) bool         { return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') }
func isIdentBegin(ch byte) bool    { return ch == '_' || ch == '$' || isAlpha(ch) }
func isIdentContinue(ch byte) bool { return isIdentBegin(ch) || isDigit(ch) }

// isRegexContext 判断是否处于可解析正则字面量的位置
func isRegexContext(tokens []Token) bool {
	if len(tokens) == 0 {
		return true
	}
	switch tokens[len(tokens)-1].Type {
	case TokIdentifier, TokNumber, TokString, TokRegex, TokNull, TokUndefined, TokBool, TokInc, TokDec, TokPlus, TokMinus, TokAsterisk, TokSlash, TokRem, TokLT, TokLTE, TokGT, TokGTE, TokLogicalAnd, TokLogicalOr, TokLogicalNot, TokBitAnd, TokBitOr, TokBitXor, TokBitNot, TokDot, TokRParen, TokRBracket, TokRBrace:
		return false
	default:
		return true
	}
}
