package imports

import (
	"reflect"
	"strings"

	"git.woa.com/vasd_masc_ba/YitihuaOteam/base/gofun/register"
)

var _ = reflect.Int

func init() {
	register.AddPackage("strings", "strings",
		register.NewType("Builder", reflect.TypeOf(func(strings.Builder) {}).In(0), ""),
		register.NewFunction("Compare", strings.Compare, ""),
		register.NewFunction("Contains", strings.Contains, ""),
		register.NewFunction("ContainsAny", strings.ContainsAny, ""),
		register.NewFunction("ContainsRune", strings.ContainsRune, ""),
		register.NewFunction("Count", strings.Count, ""),
		register.NewFunction("EqualFold", strings.EqualFold, ""),
		register.NewFunction("Fields", strings.Fields, ""),
		register.NewFunction("FieldsFunc", strings.FieldsFunc, ""),
		register.NewFunction("HasPrefix", strings.HasPrefix, ""),
		register.NewFunction("HasSuffix", strings.HasSuffix, ""),
		register.NewFunction("Index", strings.Index, ""),
		register.NewFunction("IndexAny", strings.IndexAny, ""),
		register.NewFunction("IndexByte", strings.IndexByte, ""),
		register.NewFunction("IndexFunc", strings.IndexFunc, ""),
		register.NewFunction("IndexRune", strings.IndexRune, ""),
		register.NewFunction("Join", strings.Join, ""),
		register.NewFunction("LastIndex", strings.LastIndex, ""),
		register.NewFunction("LastIndexAny", strings.LastIndexAny, ""),
		register.NewFunction("LastIndexByte", strings.LastIndexByte, ""),
		register.NewFunction("LastIndexFunc", strings.LastIndexFunc, ""),
		register.NewFunction("Map", strings.Map, ""),
		register.NewFunction("NewReader", strings.NewReader, ""),
		register.NewFunction("NewReplacer", strings.NewReplacer, ""),
		register.NewType("Reader", reflect.TypeOf(func(strings.Reader) {}).In(0), ""),
		register.NewFunction("Repeat", strings.Repeat, ""),
		register.NewFunction("Replace", strings.Replace, ""),
		register.NewFunction("ReplaceAll", strings.ReplaceAll, ""),
		register.NewType("Replacer", reflect.TypeOf(func(strings.Replacer) {}).In(0), ""),
		register.NewFunction("Split", strings.Split, ""),
		register.NewFunction("SplitAfter", strings.SplitAfter, ""),
		register.NewFunction("SplitAfterN", strings.SplitAfterN, ""),
		register.NewFunction("SplitN", strings.SplitN, ""),
		register.NewFunction("Title", strings.Title, ""),
		register.NewFunction("ToLower", strings.ToLower, ""),
		register.NewFunction("ToLowerSpecial", strings.ToLowerSpecial, ""),
		register.NewFunction("ToTitle", strings.ToTitle, ""),
		register.NewFunction("ToTitleSpecial", strings.ToTitleSpecial, ""),
		register.NewFunction("ToUpper", strings.ToUpper, ""),
		register.NewFunction("ToUpperSpecial", strings.ToUpperSpecial, ""),
		register.NewFunction("ToValidUTF8", strings.ToValidUTF8, ""),
		register.NewFunction("Trim", strings.Trim, ""),
		register.NewFunction("TrimFunc", strings.TrimFunc, ""),
		register.NewFunction("TrimLeft", strings.TrimLeft, ""),
		register.NewFunction("TrimLeftFunc", strings.TrimLeftFunc, ""),
		register.NewFunction("TrimPrefix", strings.TrimPrefix, ""),
		register.NewFunction("TrimRight", strings.TrimRight, ""),
		register.NewFunction("TrimRightFunc", strings.TrimRightFunc, ""),
		register.NewFunction("TrimSpace", strings.TrimSpace, ""),
		register.NewFunction("TrimSuffix", strings.TrimSuffix, ""),
	)
}
