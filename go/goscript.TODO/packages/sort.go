package imports

import (
	"reflect"
	"sort"

	"git.woa.com/vasd_masc_ba/YitihuaOteam/base/gofun/register"
)

var _ = reflect.Int

func init() {
	register.AddPackage("sort", "sort",
		register.NewType("Float64Slice", reflect.TypeOf(func(sort.Float64Slice) {}).In(0), ""),
		register.NewFunction("Float64s", sort.Float64s, ""),
		register.NewFunction("Float64sAreSorted", sort.Float64sAreSorted, ""),
		register.NewType("IntSlice", reflect.TypeOf(func(sort.IntSlice) {}).In(0), ""),
		register.NewType("Interface", reflect.TypeOf(func(sort.Interface) {}).In(0), ""),
		register.NewFunction("Ints", sort.Ints, ""),
		register.NewFunction("IntsAreSorted", sort.IntsAreSorted, ""),
		register.NewFunction("IsSorted", sort.IsSorted, ""),
		register.NewFunction("Reverse", sort.Reverse, ""),
		register.NewFunction("Search", sort.Search, ""),
		register.NewFunction("SearchFloat64s", sort.SearchFloat64s, ""),
		register.NewFunction("SearchInts", sort.SearchInts, ""),
		register.NewFunction("SearchStrings", sort.SearchStrings, ""),
		register.NewFunction("Slice", sort.Slice, ""),
		register.NewFunction("SliceIsSorted", sort.SliceIsSorted, ""),
		register.NewFunction("SliceStable", sort.SliceStable, ""),
		register.NewFunction("Sort", sort.Sort, ""),
		register.NewFunction("Stable", sort.Stable, ""),
		register.NewType("StringSlice", reflect.TypeOf(func(sort.StringSlice) {}).In(0), ""),
		register.NewFunction("Strings", sort.Strings, ""),
		register.NewFunction("StringsAreSorted", sort.StringsAreSorted, ""),
	)
}
