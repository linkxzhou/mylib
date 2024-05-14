package external_imports

import (
	"reflect"

	"git.woa.com/vasd_masc_ba/YitihuaOteam/base/tnm2"

	"git.woa.com/vasd_masc_ba/YitihuaOteam/base/gofun/register"
)

var _ = reflect.Int

func init() {
	register.AddPackage("git.woa.com/vasd_masc_ba/YitihuaOteam/base/tnm2", "tnm2",
		register.NewFunction("AttrApi", tnm2.AttrApi, ""),
		register.NewFunction("AttrApiSet", tnm2.AttrApiSet, ""),
		register.NewType("Attrs", reflect.TypeOf(func(tnm2.Attrs) {}).In(0), ""),
		register.NewVar("ErrAlreadyAllocate", &tnm2.ErrAlreadyAllocate, reflect.TypeOf(func(error) {}).In(0), ""),
		register.NewVar("ErrNotAllocate", &tnm2.ErrNotAllocate, reflect.TypeOf(func(error) {}).In(0), ""),
		register.NewVar("ErrNotFound", &tnm2.ErrNotFound, reflect.TypeOf(func(error) {}).In(0), ""),
		register.NewVar("ErrUnknown", &tnm2.ErrUnknown, reflect.TypeOf(func(error) {}).In(0), ""),
		register.NewFunction("GetAttrValue", tnm2.GetAttrValue, ""),
	)
}
