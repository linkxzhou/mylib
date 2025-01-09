package imports

import (
	"errors"
	"reflect"

	"git.woa.com/vasd_masc_ba/YitihuaOteam/base/gofun/register"
)

var _ = reflect.Int

func init() {
	register.AddPackage("errors", "errors",
		register.NewFunction("As", errors.As, ""),
		register.NewFunction("Is", errors.Is, ""),
		register.NewFunction("New", errors.New, ""),
		register.NewFunction("Unwrap", errors.Unwrap, ""),
	)
}
