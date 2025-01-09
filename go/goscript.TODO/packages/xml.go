package imports

import (
	"encoding/xml"
	"reflect"

	"git.woa.com/vasd_masc_ba/YitihuaOteam/base/gofun/register"
)

var _ = reflect.Int

func init() {
	register.AddPackage("encoding/xml", "xml",
		register.NewType("Attr", reflect.TypeOf(func(xml.Attr) {}).In(0), ""),
		register.NewType("CharData", reflect.TypeOf(func(xml.CharData) {}).In(0), ""),
		register.NewType("Comment", reflect.TypeOf(func(xml.Comment) {}).In(0), ""),
		register.NewFunction("CopyToken", xml.CopyToken, ""),
		register.NewType("Decoder", reflect.TypeOf(func(xml.Decoder) {}).In(0), ""),
		register.NewType("Directive", reflect.TypeOf(func(xml.Directive) {}).In(0), ""),
		register.NewType("Encoder", reflect.TypeOf(func(xml.Encoder) {}).In(0), ""),
		register.NewType("EndElement", reflect.TypeOf(func(xml.EndElement) {}).In(0), ""),
		register.NewFunction("Escape", xml.Escape, ""),
		register.NewFunction("EscapeText", xml.EscapeText, ""),
		register.NewVar("HTMLAutoClose", &xml.HTMLAutoClose, reflect.TypeOf(xml.HTMLAutoClose), ""),
		register.NewVar("HTMLEntity", &xml.HTMLEntity, reflect.TypeOf(xml.HTMLEntity), ""),
		register.NewConst("Header", xml.Header, ""),
		register.NewFunction("Marshal", xml.Marshal, ""),
		register.NewFunction("MarshalIndent", xml.MarshalIndent, ""),
		register.NewType("Marshaler", reflect.TypeOf(func(xml.Marshaler) {}).In(0), ""),
		register.NewType("MarshalerAttr", reflect.TypeOf(func(xml.MarshalerAttr) {}).In(0), ""),
		register.NewType("Name", reflect.TypeOf(func(xml.Name) {}).In(0), ""),
		register.NewFunction("NewDecoder", xml.NewDecoder, ""),
		register.NewFunction("NewEncoder", xml.NewEncoder, ""),
		register.NewFunction("NewTokenDecoder", xml.NewTokenDecoder, ""),
		register.NewType("ProcInst", reflect.TypeOf(func(xml.ProcInst) {}).In(0), ""),
		register.NewType("StartElement", reflect.TypeOf(func(xml.StartElement) {}).In(0), ""),
		register.NewType("SyntaxError", reflect.TypeOf(func(xml.SyntaxError) {}).In(0), ""),
		register.NewType("TagPathError", reflect.TypeOf(func(xml.TagPathError) {}).In(0), ""),
		register.NewType("Token", reflect.TypeOf(func(xml.Token) {}).In(0), ""),
		register.NewType("TokenReader", reflect.TypeOf(func(xml.TokenReader) {}).In(0), ""),
		register.NewFunction("Unmarshal", xml.Unmarshal, ""),
		register.NewType("UnmarshalError", reflect.TypeOf(func(xml.UnmarshalError) {}).In(0), ""),
		register.NewType("Unmarshaler", reflect.TypeOf(func(xml.Unmarshaler) {}).In(0), ""),
		register.NewType("UnmarshalerAttr", reflect.TypeOf(func(xml.UnmarshalerAttr) {}).In(0), ""),
		register.NewType("UnsupportedTypeError", reflect.TypeOf(func(xml.UnsupportedTypeError) {}).In(0), ""),
	)
}
