package imports

import (
	"crypto/des"
	"reflect"

	"git.woa.com/vasd_masc_ba/YitihuaOteam/base/gofun/register"
)

var _ = reflect.Int

func init() {
	register.AddPackage("crypto/des", "des",
		register.NewConst("BlockSize", des.BlockSize, ""),
		register.NewType("KeySizeError", reflect.TypeOf(func(des.KeySizeError) {}).In(0), ""),
		register.NewFunction("NewCipher", des.NewCipher, ""),
		register.NewFunction("NewTripleDESCipher", des.NewTripleDESCipher, ""),
	)
}
