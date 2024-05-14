package external_imports

import (
	"reflect"

	"git.woa.com/vasd_masc_ba/YitihuaOteam/base/jsonpb"

	"git.woa.com/vasd_masc_ba/YitihuaOteam/base/gofun/register"
)

var _ = reflect.Int

func init() {
	register.AddPackage("git.woa.com/vasd_masc_ba/YitihuaOteam/base/jsonpb", "jsonpb",
		register.NewType("JsonPbObject", reflect.TypeOf(func(jsonpb.JsonPbObject) {}).In(0), ""),
		register.NewFunction("NewJsonPbObject", jsonpb.NewJsonPbObject, ""),
	)
}
