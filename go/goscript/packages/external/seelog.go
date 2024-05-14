package external_imports

import (
	"reflect"

	"git.woa.com/vasd_masc_ba/YitihuaOteam/base/log"

	"git.woa.com/vasd_masc_ba/YitihuaOteam/base/gofun/register"
)

var _ = reflect.Int

func init() {
	register.AddPackage("log", "log",
		register.NewFunction("Debug", log.Debug, ""),
		register.NewFunction("Debugf", log.Debugf, ""),
		register.NewFunction("Error", log.Error, ""),
		register.NewFunction("Errorf", log.Errorf, ""),
		register.NewFunction("Info", log.Debug, ""),
		register.NewFunction("Infof", log.Debugf, ""),
		register.NewFunction("Warn", log.Warn, ""),
		register.NewFunction("Warnf", log.Warnf, ""),
	)
}
