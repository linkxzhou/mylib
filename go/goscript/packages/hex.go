package imports

import (
	"encoding/hex"
	"reflect"

	"git.woa.com/vasd_masc_ba/YitihuaOteam/base/gofun/register"
)

var _ = reflect.Int

func init() {
	register.AddPackage("encoding/hex", "hex",
		register.NewFunction("Decode", hex.Decode, ""),
		register.NewFunction("DecodeString", hex.DecodeString, ""),
		register.NewFunction("DecodedLen", hex.DecodedLen, ""),
		register.NewFunction("Dump", hex.Dump, ""),
		register.NewFunction("Dumper", hex.Dumper, ""),
		register.NewFunction("Encode", hex.Encode, ""),
		register.NewFunction("EncodeToString", hex.EncodeToString, ""),
		register.NewFunction("EncodedLen", hex.EncodedLen, ""),
		register.NewVar("ErrLength", &hex.ErrLength, reflect.TypeOf(func(error) {}).In(0), ""),
		register.NewType("InvalidByteError", reflect.TypeOf(func(hex.InvalidByteError) {}).In(0), ""),
		register.NewFunction("NewDecoder", hex.NewDecoder, ""),
		register.NewFunction("NewEncoder", hex.NewEncoder, ""),
	)
}
