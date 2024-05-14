package imports

import (
	"container/heap"
	"reflect"

	"git.woa.com/vasd_masc_ba/YitihuaOteam/base/gofun/register"
)

var _ = reflect.Int

func init() {
	register.AddPackage("container/heap", "heap",
		register.NewFunction("Fix", heap.Fix, ""),
		register.NewFunction("Init", heap.Init, ""),
		register.NewType("Interface", reflect.TypeOf(func(heap.Interface) {}).In(0), ""),
		register.NewFunction("Pop", heap.Pop, ""),
		register.NewFunction("Push", heap.Push, ""),
		register.NewFunction("Remove", heap.Remove, ""),
	)
}
