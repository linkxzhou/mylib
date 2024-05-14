package imports

import (
	"reflect"
	"unicode/utf8"

	"git.woa.com/vasd_masc_ba/YitihuaOteam/base/gofun/register"
)

var _ = reflect.Int

func init() {
	register.AddPackage("unicode/utf8", "utf8",
		register.NewFunction("DecodeLastRune", utf8.DecodeLastRune, ""),
		register.NewFunction("DecodeLastRuneInString", utf8.DecodeLastRuneInString, ""),
		register.NewFunction("DecodeRune", utf8.DecodeRune, ""),
		register.NewFunction("DecodeRuneInString", utf8.DecodeRuneInString, ""),
		register.NewFunction("EncodeRune", utf8.EncodeRune, ""),
		register.NewFunction("FullRune", utf8.FullRune, ""),
		register.NewFunction("FullRuneInString", utf8.FullRuneInString, ""),
		register.NewConst("MaxRune", utf8.MaxRune, ""),
		register.NewFunction("RuneCount", utf8.RuneCount, ""),
		register.NewFunction("RuneCountInString", utf8.RuneCountInString, ""),
		register.NewConst("RuneError", utf8.RuneError, ""),
		register.NewFunction("RuneLen", utf8.RuneLen, ""),
		register.NewConst("RuneSelf", utf8.RuneSelf, ""),
		register.NewFunction("RuneStart", utf8.RuneStart, ""),
		register.NewConst("UTFMax", utf8.UTFMax, ""),
		register.NewFunction("Valid", utf8.Valid, ""),
		register.NewFunction("ValidRune", utf8.ValidRune, ""),
		register.NewFunction("ValidString", utf8.ValidString, ""),
	)
}
