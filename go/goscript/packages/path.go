package imports

import (
	"path"
	"reflect"

	"git.woa.com/vasd_masc_ba/YitihuaOteam/base/gofun/register"
)

var _ = reflect.Int

func init() {
	register.AddPackage("path", "path",
		register.NewFunction("Base", path.Base, ""),
		register.NewFunction("Clean", path.Clean, ""),
		register.NewFunction("Dir", path.Dir, ""),
		register.NewVar("ErrBadPattern", &path.ErrBadPattern, reflect.TypeOf(func(error) {}).In(0), ""),
		register.NewFunction("Ext", path.Ext, ""),
		register.NewFunction("IsAbs", path.IsAbs, ""),
		register.NewFunction("Join", path.Join, ""),
		register.NewFunction("Match", path.Match, ""),
		register.NewFunction("Split", path.Split, ""),
	)
}
