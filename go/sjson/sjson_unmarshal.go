package sjson

import (
	"io"
)

// Unmarshal 将JSON字节切片直接解码到Go对象
func Unmarshal(data []byte, v interface{}) error {
	return UnmarshalWithConfig(data, v, defaultConfig)
}

// UnmarshalWithConfig 将JSON字节切片直接解码到Go对象，使用指定配置
func UnmarshalWithConfig(data []byte, v interface{}, config Config) error {
	decoder := newDecoder(string(data), config)
	return decoder.Decode(v)
}

// UnmarshalFromReader 从io.Reader读取JSON并直接解码到Go对象
func UnmarshalFromReader(r io.Reader, v interface{}) error {
	return UnmarshalFromReaderWithConfig(r, v, defaultConfig)
}

// UnmarshalFromReaderWithConfig 从io.Reader读取JSON并直接解码到Go对象，使用指定配置
func UnmarshalFromReaderWithConfig(r io.Reader, v interface{}, config Config) error {
	decoder, err := newDecoderFromReader(r, config)
	if err != nil {
		return err
	}
	return decoder.Decode(v)
}
