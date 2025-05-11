package sjson

import (
	"fmt"
	"io"
	"reflect"
	"sync"
)

var (
	trueValue  = reflect.ValueOf(true)
	falseValue = reflect.ValueOf(false)
)

// 用于缓存反射值的对象池
var valueSlicePool = sync.Pool{
	New: func() interface{} {
		// 预分配8个元素的容量，这是一个平衡点
		s := make([]reflect.Value, 0, 8)
		return &s
	},
}

// 用于缓存接口数组的对象池
var interfaceSlicePool = sync.Pool{
	New: func() interface{} {
		s := make([]interface{}, 0, 8)
		return &s
	},
}

// Decoder 直接从JSON文本解码到Go对象，无需中间Value对象
type Decoder struct {
	lexer  *Lexer
	token  Token
	config Config
}

// 创建新的直接解码器
func newDecoder(input []byte, config Config) *Decoder {
	lexer := NewLexer(input)
	d := &Decoder{
		lexer:  lexer,
		config: config,
	}
	d.nextToken() // 读取第一个token
	return d
}

// 从io.Reader创建新的直接解码器
func newDecoderFromReader(r io.Reader, config Config) (*Decoder, error) {
	lexer, err := NewLexerFromReader(r)
	if err != nil {
		return nil, err
	}

	d := &Decoder{
		lexer:  lexer,
		config: config,
	}
	d.nextToken() // 读取第一个token
	return d, nil
}

// 读取下一个token
func (d *Decoder) nextToken() {
	d.token = d.lexer.NextToken()
}

// 直接解码到目标对象
func (d *Decoder) Decode(v interface{}) error {
	rv := reflect.ValueOf(v)
	if rv.Kind() != reflect.Ptr || rv.IsNil() {
		return fmt.Errorf("解码目标必须是非nil指针")
	}

	// 解码值到指针所指向的对象
	if err := d.decodeValue(rv); err != nil {
		return err
	}

	// 确保解析完毕，没有多余的token
	if d.token.Type != EOFToken {
		return fmt.Errorf("JSON解析完成后存在多余内容: %v", d.token)
	}

	return nil
}
