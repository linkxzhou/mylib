package sjson

import (
	"reflect"
	"sync"
)

// 编码器流对象池，用于减少内存分配
type encoderStream struct {
	buffer []byte
}

var encoderStreamPool = sync.Pool{
	New: func() interface{} {
		stream := &encoderStream{
			buffer: make([]byte, 0, 1024),
		}
		return stream
	},
}

// 获取一个编码器流
func getEncoderStream() *encoderStream {
	return encoderStreamPool.Get().(*encoderStream)
}

// 释放一个编码器流
func releaseEncoderStream(stream *encoderStream) {
	stream.buffer = stream.buffer[:0]
	encoderStreamPool.Put(stream)
}

// Marshal 使用直接编码模式将Go对象编码为JSON字节切片
func Marshal(v interface{}) ([]byte, error) {
	// 从对象池获取编码器流
	stream := getEncoderStream()
	defer releaseEncoderStream(stream)

	// 保存编码后的结果
	err := encodeValueToBytes(stream, reflect.ValueOf(v), reflect.TypeOf(v))
	if err != nil {
		return nil, err
	}

	result := append([]byte(nil), stream.buffer...)
	return result, nil
}

// MarshalString 使用直接编码模式将Go对象编码为JSON字符串
func MarshalString(v interface{}) (string, error) {
	// 复用 Marshal 函数并转换为字符串
	bytes, err := Marshal(v)
	if err != nil {
		return "", err
	}
	return bytesToString(bytes), nil
}
