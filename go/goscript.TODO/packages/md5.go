package imports

import (
	"crypto/md5"
	"reflect"

	"git.woa.com/vasd_masc_ba/YitihuaOteam/base/gofun/register"
)

var _ = reflect.Int

func init() {
	register.AddPackage("crypto/md5", "md5",
		register.NewConst("BlockSize", md5.BlockSize, ""),
		register.NewFunction("New", md5.New, ""),
		register.NewConst("Size", md5.Size, ""),
		register.NewFunction("Sum", md5.Sum, ""),
	)
}
