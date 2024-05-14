package imports

import (
	"container/ring"
	"reflect"

	"git.woa.com/vasd_masc_ba/YitihuaOteam/base/gofun/register"
)

var _ = reflect.Int

func init() {
	register.AddPackage("container/ring", "ring",
		register.NewFunction("New", ring.New, ""),
		register.NewType("Ring", reflect.TypeOf(func(ring.Ring) {}).In(0), ""),
	)
}
