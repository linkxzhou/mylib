package sjson

import (
	"fmt"
)

// 跳过一个JSON值
func (d *Decoder) skipValue() error {
	switch d.token.Type {
	case NullToken, TrueToken, FalseToken, NumberToken, StringToken:
		// 简单值，直接跳过
		d.nextToken()
		return nil

	case LeftBraceToken:
		// 跳过对象
		return d.skipObject()

	case LeftBracketToken:
		// 跳过数组
		return d.skipArray()

	default:
		return fmt.Errorf("无法跳过未知的JSON标记: %v", d.token)
	}
}

// 跳过对象
func (d *Decoder) skipObject() error {
	// 跳过左大括号
	d.nextToken()

	// 空对象快速处理
	if d.token.Type == RightBraceToken {
		d.nextToken()
		return nil
	}

	// 跳过所有键值对
	for {
		// 跳过键
		if d.token.Type != StringToken {
			return fmt.Errorf("对象键必须是字符串，得到: %v", d.token)
		}
		d.nextToken()

		// 跳过冒号
		if d.token.Type != ColonToken {
			return fmt.Errorf("对象键后面必须是冒号，得到: %v", d.token)
		}
		d.nextToken()

		// 跳过值
		if err := d.skipValue(); err != nil {
			return err
		}

		// 检查分隔符
		if d.token.Type == CommaToken {
			d.nextToken()
		} else if d.token.Type == RightBraceToken {
			d.nextToken()
			break
		} else {
			return fmt.Errorf("对象中意外的标记: %v", d.token)
		}
	}

	return nil
}

// 跳过数组
func (d *Decoder) skipArray() error {
	// 跳过左方括号
	d.nextToken()

	// 空数组快速处理
	if d.token.Type == RightBracketToken {
		d.nextToken()
		return nil
	}

	// 跳过所有元素
	for {
		// 跳过值
		if err := d.skipValue(); err != nil {
			return err
		}

		// 检查分隔符
		if d.token.Type == CommaToken {
			d.nextToken()
		} else if d.token.Type == RightBracketToken {
			d.nextToken()
			break
		} else {
			return fmt.Errorf("数组中意外的标记: %v", d.token)
		}
	}

	return nil
}
