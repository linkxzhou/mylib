package simplejs

import (
	"fmt"
	"io"
)

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
