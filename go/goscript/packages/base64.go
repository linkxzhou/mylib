package imports

import (
	"encoding/base64"
	"reflect"

	"git.woa.com/vasd_masc_ba/YitihuaOteam/base/gofun/register"
)

var _ = reflect.Int

func init() {
	register.AddPackage("encoding/base64", "base64",
		register.NewType("CorruptInputError", reflect.TypeOf(func(base64.CorruptInputError) {}).In(0), ""),
		register.NewType("Encoding", reflect.TypeOf(func(base64.Encoding) {}).In(0), ""),
		register.NewFunction("NewDecoder", base64.NewDecoder, ""),
		register.NewFunction("NewEncoder", base64.NewEncoder, ""),
		register.NewFunction("NewEncoding", base64.NewEncoding, ""),
		register.NewConst("NoPadding", base64.NoPadding, ""),
		register.NewVar("RawStdEncoding", &base64.RawStdEncoding, reflect.TypeOf(base64.RawStdEncoding), ""),
		register.NewVar("RawURLEncoding", &base64.RawURLEncoding, reflect.TypeOf(base64.RawURLEncoding), ""),
		register.NewVar("StdEncoding", &base64.StdEncoding, reflect.TypeOf(base64.StdEncoding), ""),
		register.NewConst("StdPadding", base64.StdPadding, ""),
		register.NewVar("URLEncoding", &base64.URLEncoding, reflect.TypeOf(base64.URLEncoding), ""),
	)
}
