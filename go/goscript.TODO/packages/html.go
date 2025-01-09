package imports

import (
	"html"
	"reflect"

	"git.woa.com/vasd_masc_ba/YitihuaOteam/base/gofun/register"
)

var _ = reflect.Int

func init() {
	register.AddPackage("html", "html",
		register.NewFunction("EscapeString", html.EscapeString, ""),
		register.NewFunction("UnescapeString", html.UnescapeString, ""),
	)
}
