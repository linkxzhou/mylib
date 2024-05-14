package imports

import (
	"math/rand"
	"reflect"

	"git.woa.com/vasd_masc_ba/YitihuaOteam/base/gofun/register"
)

var _ = reflect.Int

func init() {
	register.AddPackage("math/rand", "rand",
		register.NewFunction("ExpFloat64", rand.ExpFloat64, ""),
		register.NewFunction("Float32", rand.Float32, ""),
		register.NewFunction("Float64", rand.Float64, ""),
		register.NewFunction("Int", rand.Int, ""),
		register.NewFunction("Int31", rand.Int31, ""),
		register.NewFunction("Int31n", rand.Int31n, ""),
		register.NewFunction("Int63", rand.Int63, ""),
		register.NewFunction("Int63n", rand.Int63n, ""),
		register.NewFunction("Intn", rand.Intn, ""),
		register.NewFunction("New", rand.New, ""),
		register.NewFunction("NewSource", rand.NewSource, ""),
		register.NewFunction("NewZipf", rand.NewZipf, ""),
		register.NewFunction("NormFloat64", rand.NormFloat64, ""),
		register.NewFunction("Perm", rand.Perm, ""),
		register.NewType("Rand", reflect.TypeOf(func(rand.Rand) {}).In(0), ""),
		register.NewFunction("Read", rand.Read, ""),
		register.NewFunction("Seed", rand.Seed, ""),
		register.NewFunction("Shuffle", rand.Shuffle, ""),
		register.NewType("Source", reflect.TypeOf(func(rand.Source) {}).In(0), ""),
		register.NewType("Source64", reflect.TypeOf(func(rand.Source64) {}).In(0), ""),
		register.NewFunction("Uint32", rand.Uint32, ""),
		register.NewFunction("Uint64", rand.Uint64, ""),
		register.NewType("Zipf", reflect.TypeOf(func(rand.Zipf) {}).In(0), ""),
	)
}
