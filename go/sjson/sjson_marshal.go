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

	// 预分配一定容量，但不改变内容
	if v != nil && cap(stream.buffer) < 512 {
		// 创建一个更大容量的buffer
		newBuffer := make([]byte, len(stream.buffer), 512)
		copy(newBuffer, stream.buffer)
		stream.buffer = newBuffer
	}

	// 保存编码后的结果
	var err error
	stream.buffer, err = encodeValueToBytes(stream.buffer, reflect.ValueOf(v))
	if err != nil {
		releaseEncoderStream(stream)
		return nil, err
	}

	// 创建副本并返回结果
	result := make([]byte, len(stream.buffer))
	copy(result, stream.buffer)

	releaseEncoderStream(stream)
	return result, nil
}

// MarshalString 使用直接编码模式将Go对象编码为JSON字符串
func MarshalString(v interface{}) (string, error) {
	// 复用 Marshal 函数并转换为字符串
	bytes, err := Marshal(v)
	if err != nil {
		return "", err
	}
	return string(bytes), nil
}
