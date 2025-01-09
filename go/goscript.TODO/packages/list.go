package imports

import (
	"container/list"
	"reflect"

	"git.woa.com/vasd_masc_ba/YitihuaOteam/base/gofun/register"
)

var _ = reflect.Int

func init() {
	register.AddPackage("container/list", "list",
		register.NewType("Element", reflect.TypeOf(func(list.Element) {}).In(0), ""),
		register.NewType("List", reflect.TypeOf(func(list.List) {}).In(0), ""),
		register.NewFunction("New", list.New, ""),
	)
}
