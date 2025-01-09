package imports

import (
	"bytes"
	"reflect"

	"git.woa.com/vasd_masc_ba/YitihuaOteam/base/gofun/register"
)

var _ = reflect.Int

func init() {
	register.AddPackage("bytes", "bytes",
		register.NewType("Buffer", reflect.TypeOf(func(bytes.Buffer) {}).In(0), ""),
		register.NewFunction("Compare", bytes.Compare, ""),
		register.NewFunction("Contains", bytes.Contains, ""),
		register.NewFunction("ContainsAny", bytes.ContainsAny, ""),
		register.NewFunction("ContainsRune", bytes.ContainsRune, ""),
		register.NewFunction("Count", bytes.Count, ""),
		register.NewFunction("Equal", bytes.Equal, ""),
		register.NewFunction("EqualFold", bytes.EqualFold, ""),
		register.NewVar("ErrTooLarge", &bytes.ErrTooLarge, reflect.TypeOf(func(error) {}).In(0), ""),
		register.NewFunction("Fields", bytes.Fields, ""),
		register.NewFunction("FieldsFunc", bytes.FieldsFunc, ""),
		register.NewFunction("HasPrefix", bytes.HasPrefix, ""),
		register.NewFunction("HasSuffix", bytes.HasSuffix, ""),
		register.NewFunction("Index", bytes.Index, ""),
		register.NewFunction("IndexAny", bytes.IndexAny, ""),
		register.NewFunction("IndexByte", bytes.IndexByte, ""),
		register.NewFunction("IndexFunc", bytes.IndexFunc, ""),
		register.NewFunction("IndexRune", bytes.IndexRune, ""),
		register.NewFunction("Join", bytes.Join, ""),
		register.NewFunction("LastIndex", bytes.LastIndex, ""),
		register.NewFunction("LastIndexAny", bytes.LastIndexAny, ""),
		register.NewFunction("LastIndexByte", bytes.LastIndexByte, ""),
		register.NewFunction("LastIndexFunc", bytes.LastIndexFunc, ""),
		register.NewFunction("Map", bytes.Map, ""),
		register.NewConst("MinRead", bytes.MinRead, ""),
		register.NewFunction("NewBuffer", bytes.NewBuffer, ""),
		register.NewFunction("NewBufferString", bytes.NewBufferString, ""),
		register.NewFunction("NewReader", bytes.NewReader, ""),
		register.NewType("Reader", reflect.TypeOf(func(bytes.Reader) {}).In(0), ""),
		register.NewFunction("Repeat", bytes.Repeat, ""),
		register.NewFunction("Replace", bytes.Replace, ""),
		register.NewFunction("ReplaceAll", bytes.ReplaceAll, ""),
		register.NewFunction("Runes", bytes.Runes, ""),
		register.NewFunction("Split", bytes.Split, ""),
		register.NewFunction("SplitAfter", bytes.SplitAfter, ""),
		register.NewFunction("SplitAfterN", bytes.SplitAfterN, ""),
		register.NewFunction("SplitN", bytes.SplitN, ""),
		register.NewFunction("Title", bytes.Title, ""),
		register.NewFunction("ToLower", bytes.ToLower, ""),
		register.NewFunction("ToLowerSpecial", bytes.ToLowerSpecial, ""),
		register.NewFunction("ToTitle", bytes.ToTitle, ""),
		register.NewFunction("ToTitleSpecial", bytes.ToTitleSpecial, ""),
		register.NewFunction("ToUpper", bytes.ToUpper, ""),
		register.NewFunction("ToUpperSpecial", bytes.ToUpperSpecial, ""),
		register.NewFunction("ToValidUTF8", bytes.ToValidUTF8, ""),
		register.NewFunction("Trim", bytes.Trim, ""),
		register.NewFunction("TrimFunc", bytes.TrimFunc, ""),
		register.NewFunction("TrimLeft", bytes.TrimLeft, ""),
		register.NewFunction("TrimLeftFunc", bytes.TrimLeftFunc, ""),
		register.NewFunction("TrimPrefix", bytes.TrimPrefix, ""),
		register.NewFunction("TrimRight", bytes.TrimRight, ""),
		register.NewFunction("TrimRightFunc", bytes.TrimRightFunc, ""),
		register.NewFunction("TrimSpace", bytes.TrimSpace, ""),
		register.NewFunction("TrimSuffix", bytes.TrimSuffix, ""),
	)
}
