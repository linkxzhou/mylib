package imports

import (
	"net/url"
	"reflect"

	"git.woa.com/vasd_masc_ba/YitihuaOteam/base/gofun/register"
)

var _ = reflect.Int

func init() {
	register.AddPackage("net/url", "url",
		register.NewType("Error", reflect.TypeOf(func(url.Error) {}).In(0), ""),
		register.NewType("EscapeError", reflect.TypeOf(func(url.EscapeError) {}).In(0), ""),
		register.NewType("InvalidHostError", reflect.TypeOf(func(url.InvalidHostError) {}).In(0), ""),
		register.NewFunction("Parse", url.Parse, ""),
		register.NewFunction("ParseQuery", url.ParseQuery, ""),
		register.NewFunction("ParseRequestURI", url.ParseRequestURI, ""),
		register.NewFunction("PathEscape", url.PathEscape, ""),
		register.NewFunction("PathUnescape", url.PathUnescape, ""),
		register.NewFunction("QueryEscape", url.QueryEscape, ""),
		register.NewFunction("QueryUnescape", url.QueryUnescape, ""),
		register.NewType("URL", reflect.TypeOf(func(url.URL) {}).In(0), ""),
		register.NewFunction("User", url.User, ""),
		register.NewFunction("UserPassword", url.UserPassword, ""),
		register.NewType("Userinfo", reflect.TypeOf(func(url.Userinfo) {}).In(0), ""),
		register.NewType("Values", reflect.TypeOf(func(url.Values) {}).In(0), ""),
	)
}
