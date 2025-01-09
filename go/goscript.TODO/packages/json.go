package imports

import (
	"encoding/json"
	"reflect"

	"git.woa.com/vasd_masc_ba/YitihuaOteam/base/gofun/register"
)

var _ = reflect.Int

func init() {
	register.AddPackage("encoding/json", "json",
		register.NewFunction("Compact", json.Compact, ""),
		register.NewType("Decoder", reflect.TypeOf(func(json.Decoder) {}).In(0), ""),
		register.NewType("Delim", reflect.TypeOf(func(json.Delim) {}).In(0), ""),
		register.NewType("Encoder", reflect.TypeOf(func(json.Encoder) {}).In(0), ""),
		register.NewFunction("HTMLEscape", json.HTMLEscape, ""),
		register.NewFunction("Indent", json.Indent, ""),
		register.NewType("InvalidUTF8Error", reflect.TypeOf(func(json.InvalidUTF8Error) {}).In(0), ""),
		register.NewType("InvalidUnmarshalError", reflect.TypeOf(func(json.InvalidUnmarshalError) {}).In(0), ""),
		register.NewFunction("Marshal", json.Marshal, ""),
		register.NewFunction("MarshalIndent", json.MarshalIndent, ""),
		register.NewType("Marshaler", reflect.TypeOf(func(json.Marshaler) {}).In(0), ""),
		register.NewType("MarshalerError", reflect.TypeOf(func(json.MarshalerError) {}).In(0), ""),
		register.NewFunction("NewDecoder", json.NewDecoder, ""),
		register.NewFunction("NewEncoder", json.NewEncoder, ""),
		register.NewType("Number", reflect.TypeOf(func(json.Number) {}).In(0), ""),
		register.NewType("RawMessage", reflect.TypeOf(func(json.RawMessage) {}).In(0), ""),
		register.NewType("SyntaxError", reflect.TypeOf(func(json.SyntaxError) {}).In(0), ""),
		register.NewType("Token", reflect.TypeOf(func(json.Token) {}).In(0), ""),
		register.NewFunction("Unmarshal", json.Unmarshal, ""),
		register.NewType("UnmarshalFieldError", reflect.TypeOf(func(json.UnmarshalFieldError) {}).In(0), ""),
		register.NewType("UnmarshalTypeError", reflect.TypeOf(func(json.UnmarshalTypeError) {}).In(0), ""),
		register.NewType("Unmarshaler", reflect.TypeOf(func(json.Unmarshaler) {}).In(0), ""),
		register.NewType("UnsupportedTypeError", reflect.TypeOf(func(json.UnsupportedTypeError) {}).In(0), ""),
		register.NewType("UnsupportedValueError", reflect.TypeOf(func(json.UnsupportedValueError) {}).In(0), ""),
		register.NewFunction("Valid", json.Valid, ""),
	)
}
