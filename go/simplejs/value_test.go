package simplejs

import (
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestJSValueToStringPrimitives(t *testing.T) {
	// number
	num := JSValue{Type: JSNumber, Number: 3.14}
	assert.Equal(t, "3.14", num.ToString())
	// string
	str := JSValue{Type: JSString, String: "hello"}
	assert.Equal(t, "hello", str.ToString())
	// bool
	b := JSValue{Type: JSBoolean, Bool: true}
	assert.Equal(t, "true", b.ToString())
	// null
	null := JSValue{Type: JSNull}
	assert.Equal(t, "null", null.ToString())
	// undefined
	undef := JSValue{Type: JSUndefined}
	assert.Equal(t, "undefined", undef.ToString())
}

func TestJSValueToStringObjectAndArray(t *testing.T) {
	// object
	objMap := map[string]JSValue{"a": {Type: JSNumber, Number: 1}}
	obj := JSValue{Type: JSObject, Object: objMap}
	assert.Equal(t, "[object Object]", obj.ToString())
	// array
	arr := JSValue{Type: JSObject, IsArray: true, Object: []JSValue{{Type: JSNumber, Number: 1}, {Type: JSUndefined}}}
	assert.Equal(t, "[1, undefined]", arr.ToString())
}

func TestJSValueConversions(t *testing.T) {
	// ToObject
	m := map[string]JSValue{"x": {Type: JSNumber, Number: 2}}
	v := JSValue{Type: JSObject, Object: m}
	assert.Equal(t, m, v.ToObject())
	// ToArray
	a := []JSValue{{Type: JSNumber, Number: 5}}
	vArr := JSValue{Type: JSObject, Object: a}
	assert.Equal(t, a, vArr.ToArray())
	// IsArrayType
	vArr.IsArray = true
	assert.True(t, vArr.IsArrayType())
	assert.False(t, v.IsArrayType())
}

func TestJSValueFunctionAndError(t *testing.T) {
	// function
	fn := func(args ...JSValue) JSValue { return JSValue{Type: JSNumber, Number: 7} }
	vfn := JSValue{Type: JSFunction, Function: fn}
	call := vfn.ToFunction()
	assert.NotNil(t, call)
	// invoking function should return expected JSValue
	res := call()
	assert.Equal(t, JSValue{Type: JSNumber, Number: 7}, res)
	// error
	err := errors.New("fail")
	verr := JSValue{Type: JSError, Error: err}
	assert.Equal(t, err, verr.ToError())
	// String on error
	assert.Equal(t, "fail", verr.ToString())
}
