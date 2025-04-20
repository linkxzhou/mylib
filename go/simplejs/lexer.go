package simplejs

import (
	"fmt"
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

// Tokenize splits input into tokens.
func Tokenize(input string) ([]Token, error) {
	tokens := []Token{}
	lineNum := 1
	for i := 0; i < len(input); {
		ch := input[i]
		// skip whitespace
		if isSpace(ch) {
			if ch == '\n' {
				lineNum++
			}
			i++
			continue
		}
		// skip single-line comment
		if ch == '/' && i+1 < len(input) && input[i+1] == '/' {
			for i < len(input) && input[i] != '\n' {
				i++
			}
			if i < len(input) {
				lineNum++
				i++ // skip the newline
			}
			continue
		}
		// skip multi-line comment
		if ch == '/' && i+1 < len(input) && input[i+1] == '*' {
			i += 2
			for i+1 < len(input) && !(input[i] == '*' && input[i+1] == '/') {
				if input[i] == '\n' {
					lineNum++
				}
				i++
			}
			if i+1 < len(input) {
				i += 2
			}
			continue
		}
		// number literal
		if isDigit(ch) || (ch == '.' && i+1 < len(input) && isDigit(input[i+1])) {
			start := i
			for i < len(input) && (isDigit(input[i]) || input[i] == '.') {
				i++
			}
			tokens = append(tokens, Token{Type: TokNumber, Literal: input[start:i], Line: lineNum})
			continue
		}
		// identifier or keyword
		if isIdentBegin(ch) {
			start := i
			i++
			for i < len(input) && isIdentContinue(input[i]) {
				i++
			}
			lit := input[start:i]
			switch lit {
			case "true", "false":
				tokens = append(tokens, Token{Type: TokBool, Literal: lit, Line: lineNum})
			case "null":
				tokens = append(tokens, Token{Type: TokNull, Literal: lit, Line: lineNum})
			case "undefined":
				tokens = append(tokens, Token{Type: TokUndefined, Literal: lit, Line: lineNum})
			case "if":
				tokens = append(tokens, Token{Type: TokIf, Literal: lit, Line: lineNum})
			case "else":
				tokens = append(tokens, Token{Type: TokElse, Literal: lit, Line: lineNum})
			case "while":
				tokens = append(tokens, Token{Type: TokWhile, Literal: lit, Line: lineNum})
			case "for":
				tokens = append(tokens, Token{Type: TokFor, Literal: lit, Line: lineNum})
			case "function":
				tokens = append(tokens, Token{Type: TokFunction, Literal: lit, Line: lineNum})
			case "return":
				tokens = append(tokens, Token{Type: TokReturn, Literal: lit, Line: lineNum})
			case "var":
				tokens = append(tokens, Token{Type: TokVar, Literal: lit, Line: lineNum})
			case "let":
				tokens = append(tokens, Token{Type: TokLet, Literal: lit, Line: lineNum})
			case "const":
				tokens = append(tokens, Token{Type: TokConst, Literal: lit, Line: lineNum})
			case "class":
				tokens = append(tokens, Token{Type: TokClass, Literal: lit, Line: lineNum})
			case "extends":
				tokens = append(tokens, Token{Type: TokExtends, Literal: lit, Line: lineNum})
			case "new":
				tokens = append(tokens, Token{Type: TokNew, Literal: lit, Line: lineNum})
			case "super":
				tokens = append(tokens, Token{Type: TokSuper, Literal: lit, Line: lineNum})
			case "throw":
				tokens = append(tokens, Token{Type: TokThrow, Literal: lit, Line: lineNum})
			case "try":
				tokens = append(tokens, Token{Type: TokTry, Literal: lit, Line: lineNum})
			case "catch":
				tokens = append(tokens, Token{Type: TokCatch, Literal: lit, Line: lineNum})
			case "break":
				tokens = append(tokens, Token{Type: TokBreak, Literal: lit, Line: lineNum})
			case "delete":
				tokens = append(tokens, Token{Type: TokDelete, Literal: lit, Line: lineNum})
			default:
				tokens = append(tokens, Token{Type: TokIdentifier, Literal: lit, Line: lineNum})
			}
			continue
		}
		// string literal
		if ch == '"' || ch == '\'' {
			quote := ch
			start := i + 1
			i++
			for i < len(input) && input[i] != quote {
				if input[i] == '\\' && i+1 < len(input) {
					i += 2
				} else {
					i++
				}
			}
			if i >= len(input) {
				return tokens, fmt.Errorf("unterminated string")
			}
			lit := input[start:i]
			i++ // consume closing quote
			tokens = append(tokens, Token{Type: TokString, Literal: lit, Line: lineNum})
			continue
		}
		// operators and punctuation
		switch ch {
		case '(':
			tokens = append(tokens, Token{Type: TokLParen, Literal: "(", Line: lineNum})
			i++
		case ')':
			tokens = append(tokens, Token{Type: TokRParen, Literal: ")", Line: lineNum})
			i++
		case '{':
			tokens = append(tokens, Token{Type: TokLBrace, Literal: "{", Line: lineNum})
			i++
		case '}':
			tokens = append(tokens, Token{Type: TokRBrace, Literal: "}", Line: lineNum})
			i++
		case '[':
			tokens = append(tokens, Token{Type: TokLBracket, Literal: "[", Line: lineNum})
			i++
		case ']':
			tokens = append(tokens, Token{Type: TokRBracket, Literal: "]", Line: lineNum})
			i++
		case ';':
			tokens = append(tokens, Token{Type: TokSemicolon, Literal: ";", Line: lineNum})
			i++
		case ',':
			tokens = append(tokens, Token{Type: TokComma, Literal: ",", Line: lineNum})
			i++
		case '+':
			if i+1 < len(input) && input[i+1] == '=' {
				return tokens, fmt.Errorf("+= is not supported")
			} else if i+1 < len(input) && input[i+1] == '+' {
				return tokens, fmt.Errorf("++ is not supported")
			} else {
				tokens = append(tokens, Token{Type: TokPlus, Literal: "+", Line: lineNum})
				i++
			}
		case '-':
			if i+1 < len(input) && input[i+1] == '=' {
				tokens = append(tokens, Token{Type: TokMinusAssign, Literal: "-=", Line: lineNum})
				i += 2
			} else if i+1 < len(input) && input[i+1] == '-' {
				return tokens, fmt.Errorf("-- is not supported")
			} else {
				tokens = append(tokens, Token{Type: TokMinus, Literal: "-", Line: lineNum})
				i++
			}
		case '*':
			if i+1 < len(input) && input[i+1] == '=' {
				tokens = append(tokens, Token{Type: TokAsteriskAssign, Literal: "*=", Line: lineNum})
				i += 2
			} else {
				tokens = append(tokens, Token{Type: TokAsterisk, Literal: "*", Line: lineNum})
				i++
			}
		case '/':
			if i+1 < len(input) && input[i+1] == '=' {
				tokens = append(tokens, Token{Type: TokSlashAssign, Literal: "/=", Line: lineNum})
				i += 2
			} else {
				tokens = append(tokens, Token{Type: TokSlash, Literal: "/", Line: lineNum})
				i++
			}
		case '=':
			// strict equal ===
			if i+2 < len(input) && input[i+1] == '=' && input[i+2] == '=' {
				tokens = append(tokens, Token{Type: TokStrictEqual, Literal: "===", Line: lineNum})
				i += 3
				break
			}
			if i+1 < len(input) && input[i+1] == '>' {
				tokens = append(tokens, Token{Type: TokArrow, Literal: "=>", Line: lineNum})
				i += 2
			} else if i+1 < len(input) && input[i+1] == '=' {
				tokens = append(tokens, Token{Type: TokEqual, Literal: "==", Line: lineNum})
				i += 2
			} else {
				tokens = append(tokens, Token{Type: TokAssign, Literal: "=", Line: lineNum})
				i++
			}
		case '!':
			// strict not equal !==
			if i+2 < len(input) && input[i+1] == '=' && input[i+2] == '=' {
				tokens = append(tokens, Token{Type: TokNotStrictEqual, Literal: "!==", Line: lineNum})
				i += 3
			} else if i+1 < len(input) && input[i+1] == '=' {
				// not equal !=
				tokens = append(tokens, Token{Type: TokNotEqual, Literal: "!=", Line: lineNum})
				i += 2
			} else {
				tokens = append(tokens, Token{Type: TokLogicalNot, Literal: "!", Line: lineNum})
				i++
			}
		case '%':
			tokens = append(tokens, Token{Type: TokRem, Literal: "%", Line: lineNum})
			i++
		case '?':
			tokens = append(tokens, Token{Type: TokQuestion, Literal: "?", Line: lineNum})
			i++
		case ':':
			tokens = append(tokens, Token{Type: TokColon, Literal: ":", Line: lineNum})
			i++
		case '<':
			if i+1 < len(input) && input[i+1] == '=' {
				tokens = append(tokens, Token{Type: TokLTE, Literal: "<=", Line: lineNum})
				i += 2
			} else if i+1 < len(input) && input[i+1] == '<' {
				tokens = append(tokens, Token{Type: TokLShift, Literal: "<<", Line: lineNum})
				i += 2
			} else {
				tokens = append(tokens, Token{Type: TokLT, Literal: "<", Line: lineNum})
				i++
			}
		case '>':
			if i+2 < len(input) && input[i+1] == '>' && input[i+2] == '>' {
				tokens = append(tokens, Token{Type: TokURShift, Literal: ">>>", Line: lineNum})
				i += 3
			} else if i+1 < len(input) && input[i+1] == '>' {
				tokens = append(tokens, Token{Type: TokRShift, Literal: ">>", Line: lineNum})
				i += 2
			} else if i+1 < len(input) && input[i+1] == '=' {
				tokens = append(tokens, Token{Type: TokGTE, Literal: ">=", Line: lineNum})
				i += 2
			} else {
				tokens = append(tokens, Token{Type: TokGT, Literal: ">", Line: lineNum})
				i++
			}
		case '&':
			if i+1 < len(input) && input[i+1] == '&' {
				tokens = append(tokens, Token{Type: TokLogicalAnd, Literal: "&&", Line: lineNum})
				i += 2
			} else {
				tokens = append(tokens, Token{Type: TokBitAnd, Literal: "&", Line: lineNum})
				i++
			}
		case '|':
			if i+1 < len(input) && input[i+1] == '|' {
				tokens = append(tokens, Token{Type: TokLogicalOr, Literal: "||", Line: lineNum})
				i += 2
			} else {
				tokens = append(tokens, Token{Type: TokBitOr, Literal: "|", Line: lineNum})
				i++
			}
		case '^':
			tokens = append(tokens, Token{Type: TokBitXor, Literal: "^", Line: lineNum})
			i++
		case '~':
			tokens = append(tokens, Token{Type: TokBitNot, Literal: "~", Line: lineNum})
			i++
		case '.':
			tokens = append(tokens, Token{Type: TokDot, Literal: ".", Line: lineNum})
			i++
			continue
		case 'd':
			if i+5 <= len(input) && input[i:i+6] == "delete" {
				tokens = append(tokens, Token{Type: TokDelete, Literal: "delete", Line: lineNum})
				i += 6
				continue
			}
		default:
			return tokens, fmt.Errorf("unexpected character: %c", ch)
		}
	}
	tokens = append(tokens, Token{Type: TokEOF, Literal: "", Line: lineNum})
	return tokens, nil
}

// helpers
func isSpace(ch byte) bool {
	return ch == ' ' || ch == '\n' || ch == '\r' || ch == '\t' || ch == '\f' || ch == '\v'
}
func isDigit(ch byte) bool         { return ch >= '0' && ch <= '9' }
func isAlpha(ch byte) bool         { return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') }
func isIdentBegin(ch byte) bool    { return ch == '_' || ch == '$' || isAlpha(ch) }
func isIdentContinue(ch byte) bool { return isIdentBegin(ch) || isDigit(ch) }
