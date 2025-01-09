package external_imports

import (
	"reflect"

	"git.woa.com/vasd_masc_ba/YitihuaOteam/base/proto"

	"git.woa.com/vasd_masc_ba/YitihuaOteam/base/gofun/register"
)

var _ = reflect.Int

func init() {
	register.AddPackage("git.woa.com/vasd_masc_ba/YitihuaOteam/base/proto", "proto",
		register.NewType("AddrConfig", reflect.TypeOf(func(proto.AddrConfig) {}).In(0), ""),
		register.NewFunction("NewAddrConfig", proto.NewAddrConfig, ""),
	)
}
