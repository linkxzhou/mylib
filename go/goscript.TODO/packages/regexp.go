package imports

import (
	"reflect"
	"regexp"

	"git.woa.com/vasd_masc_ba/YitihuaOteam/base/gofun/register"
)

var _ = reflect.Int

func init() {
	register.AddPackage("regexp", "regexp",
		register.NewFunction("Compile", regexp.Compile, ""),
		register.NewFunction("CompilePOSIX", regexp.CompilePOSIX, ""),
		register.NewFunction("Match", regexp.Match, ""),
		register.NewFunction("MatchReader", regexp.MatchReader, ""),
		register.NewFunction("MatchString", regexp.MatchString, ""),
		register.NewFunction("MustCompile", regexp.MustCompile, ""),
		register.NewFunction("MustCompilePOSIX", regexp.MustCompilePOSIX, ""),
		register.NewFunction("QuoteMeta", regexp.QuoteMeta, ""),
		register.NewType("Regexp", reflect.TypeOf(func(regexp.Regexp) {}).In(0), ""),
	)
}
