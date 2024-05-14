package imports

import (
	"reflect"
	"unicode/utf16"

	"git.woa.com/vasd_masc_ba/YitihuaOteam/base/gofun/register"
)

var _ = reflect.Int

func init() {
	register.AddPackage("unicode/utf16", "utf16",
		register.NewFunction("Decode", utf16.Decode, ""),
		register.NewFunction("DecodeRune", utf16.DecodeRune, ""),
		register.NewFunction("Encode", utf16.Encode, ""),
		register.NewFunction("EncodeRune", utf16.EncodeRune, ""),
		register.NewFunction("IsSurrogate", utf16.IsSurrogate, ""),
	)
}
