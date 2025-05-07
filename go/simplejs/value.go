package simplejs

import (
	"math"
	"strconv"
	"strings"
)

// JSValueType represents the type of a JS value.
type JSValueType int

const (
	JSUndefined JSValueType = iota
	JSNull
	JSBoolean
	JSNumber
	JSString
	JSObject
	JSFunction
	JSError
	JSIdentifier
	JSMember // 新增成员表达式类型
	JSSuper  // 新增 super 关键字类型
)

// JSValue is a JavaScript value.
type JSValue struct {
	Type     JSValueType
	Bool     bool
	Number   float64
	String   string
	Object   interface{} // can be map[string]JSValue or JSValue for JSMember
	Function func(args ...JSValue) JSValue
	Error    error

	FuncInfo *FunctionInfo // For user-defined functions
	Proto    *JSValue      // 原型链支持
	IsArray  bool          // Array marker
}

// FunctionInfo holds user-defined function data.
type FunctionInfo struct {
	ParamNames       []string
	Body             []Token // or AST node if available
	Closure          *Scope
	IsArrow          bool
	This             JSValue
	SuperConstructor *JSValue // Added: holds superclass constructor if any
}

func (v JSValueType) String() string {
	switch v {
	case JSUndefined:
		return "undefined"
	case JSNull:
		return "null"
	case JSBoolean:
		return "boolean"
	case JSNumber:
		return "number"
	case JSString:
		return "string"
	case JSObject:
		return "object"
	case JSFunction:
		return "function"
	case JSError:
		return "error"
	case JSIdentifier:
		return "identifier"
	case JSMember:
		return "member"
	case JSSuper:
		return "super"
	default:
		return "unknown"
	}
}

// Constructors
func Undefined() JSValue          { return JSValue{Type: JSUndefined} }
func Null() JSValue               { return JSValue{Type: JSNull} }
func BoolVal(b bool) JSValue      { return JSValue{Type: JSBoolean, Bool: b} }
func NumberVal(n float64) JSValue { return JSValue{Type: JSNumber, Number: n} }
func StringVal(s string) JSValue  { return JSValue{Type: JSString, String: s} }
func ObjectVal(o map[string]JSValue) JSValue {
	// Check if this is an array-like object (all keys are numeric or 'length')
	isArray := true
	for k := range o {
		if k == "length" {
			continue
		}
		_, err := strconv.Atoi(k)
		if err != nil {
			isArray = false
			break
		}
	}
	return JSValue{Type: JSObject, Object: o, IsArray: isArray}
}
func FunctionVal(fn func(args ...JSValue) JSValue) JSValue {
	return JSValue{Type: JSFunction, Function: fn}
}
func ErrorVal(err error) JSValue { return JSValue{Type: JSError, Error: err} }

// Type checks
func (v JSValue) IsUndefined() bool { return v.Type == JSUndefined }
func (v JSValue) IsNull() bool      { return v.Type == JSNull }
func (v JSValue) IsBoolean() bool   { return v.Type == JSBoolean }
func (v JSValue) IsNumber() bool    { return v.Type == JSNumber }
func (v JSValue) IsString() bool    { return v.Type == JSString }
func (v JSValue) IsObject() bool    { return v.Type == JSObject }
func (v JSValue) IsFunction() bool  { return v.Type == JSFunction }
func (v JSValue) IsError() bool     { return v.Type == JSError }
func (v JSValue) IsMember() bool    { return v.Type == JSMember } // 新增成员表达式类型检查
func (v JSValue) IsSuper() bool     { return v.Type == JSSuper }  // 新增 super 关键字类型检查

// Accessors
func (v JSValue) ToBool() bool {
	switch v.Type {
	case JSUndefined, JSNull:
		return false
	case JSBoolean:
		return v.Bool
	case JSNumber:
		// 0 和 NaN 为 false，其他为 true
		return v.Number != 0 && !math.IsNaN(v.Number)
	case JSString:
		// 空字符串为 false，非空为 true
		return v.String != ""
	case JSObject, JSFunction:
		// 对象和函数总是 true
		return true
	default:
		return false
	}
}

func (v JSValue) ToNumber() float64 { return v.Number }

func (v JSValue) ToString() string {
	switch v.Type {
	case JSUndefined:
		return "undefined"
	case JSNull:
		return "null"
	case JSBoolean:
		if v.Bool {
			return "true"
		}
		return "false"
	case JSNumber:
		return strconv.FormatFloat(v.Number, 'f', -1, 64)
	case JSString:
		return v.String
	case JSIdentifier:
		return v.String
	case JSObject:
		if v.IsArray {
			// handle slice arrays
			if arrSlice, ok := v.Object.([]JSValue); ok {
				parts := make([]string, len(arrSlice))
				for i, e := range arrSlice {
					parts[i] = e.ToString()
				}
				return "[" + strings.Join(parts, ", ") + "]"
			}
			// handle map-based array-like objects
			if objMap, ok := v.Object.(map[string]JSValue); ok {
				if l, ok := objMap["length"]; ok && int(l.ToNumber()) == len(objMap)-1 {
					parts := make([]string, int(l.ToNumber()))
					for i := 0; i < len(parts); i++ {
						key := strconv.Itoa(i)
						if elem, ok := objMap[key]; ok {
							parts[i] = elem.ToString()
						} else {
							parts[i] = "undefined"
						}
					}
					return "[" + strings.Join(parts, ", ") + "]"
				}
			}
		}
		return "[object Object]"
	case JSFunction:
		return "[function]"
	case JSError:
		if v.Error != nil {
			return v.Error.Error()
		}
		return "Error"
	default:
		return ""
	}
}

func (v JSValue) ToObject() map[string]JSValue {
	// If array type, convert underlying slice to object map
	if v.IsArray {
		if arr, ok := v.Object.([]JSValue); ok {
			m := make(map[string]JSValue)
			for i, e := range arr {
				m[strconv.Itoa(i)] = e
			}
			m["length"] = NumberVal(float64(len(arr)))
			return m
		}
	}
	if realv, ok := v.Object.(map[string]JSValue); ok {
		return realv
	}
	return nil
}

func (v JSValue) ToArray() []JSValue {
	if realv, ok := v.Object.([]JSValue); ok {
		return realv
	}
	// Support JSObject arrays stored as map[string]JSValue with IsArray marker
	if v.Type == JSObject && v.IsArray {
		if m, ok := v.Object.(map[string]JSValue); ok {
			lengthVal, okLen := m["length"]
			length := 0
			if okLen {
				length = int(lengthVal.ToNumber())
			}
			arr := make([]JSValue, length)
			for i := 0; i < length; i++ {
				idx := strconv.Itoa(i)
				if elem, ok := m[idx]; ok {
					arr[i] = elem
				} else {
					arr[i] = Undefined()
				}
			}
			return arr
		}
	}
	return nil
}
func (v JSValue) ToFunction() func(args ...JSValue) JSValue { return v.Function }
func (v JSValue) ToError() error                            { return v.Error }

// Add a helper to check if a JSValue is an array
func (v JSValue) IsArrayType() bool {
	return v.Type == JSObject && v.IsArray
}

// JSValue implements Statement so function declarations can be statements
// Pos returns a dummy position for JSValue
func (v JSValue) Pos() (int, int) { return 0, 0 }

// stmtNode marks JSValue as a statement
func (v JSValue) stmtNode() {}
