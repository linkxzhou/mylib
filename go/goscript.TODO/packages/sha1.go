package imports

import (
	"crypto/sha1"
	"reflect"

	"git.woa.com/vasd_masc_ba/YitihuaOteam/base/gofun/register"
)

var _ = reflect.Int

func init() {
	register.AddPackage("crypto/sha1", "sha1",
		register.NewConst("BlockSize", sha1.BlockSize, ""),
		register.NewFunction("New", sha1.New, ""),
		register.NewConst("Size", sha1.Size, ""),
		register.NewFunction("Sum", sha1.Sum, ""),
	)
}
