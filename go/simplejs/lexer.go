package simplejs

import (
	"bufio"
	"fmt"
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
