package imports

import (
	"io"

	"io/ioutil"
	"reflect"

	"git.woa.com/vasd_masc_ba/YitihuaOteam/base/gofun/register"
)

var _ = reflect.Int

func init() {
	register.AddPackage("io/ioutil", "ioutil",
		register.NewVar("Discard", &ioutil.Discard, reflect.TypeOf(func(io.Writer) {}).In(0), ""),
		register.NewFunction("NopCloser", ioutil.NopCloser, ""),
		register.NewFunction("ReadAll", ioutil.ReadAll, ""),
		register.NewFunction("ReadDir", ioutil.ReadDir, ""),
		register.NewFunction("ReadFile", ioutil.ReadFile, ""),
		register.NewFunction("TempDir", ioutil.TempDir, ""),
		register.NewFunction("TempFile", ioutil.TempFile, ""),
		register.NewFunction("WriteFile", ioutil.WriteFile, ""),
	)
}
