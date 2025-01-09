package imports

import (
	"html/template"
	"reflect"

	"git.woa.com/vasd_masc_ba/YitihuaOteam/base/gofun/register"
)

var _ = reflect.Int

func init() {
	register.AddPackage("html/template", "template",
		register.NewType("CSS", reflect.TypeOf(func(template.CSS) {}).In(0), ""),
		register.NewConst("ErrAmbigContext", template.ErrAmbigContext, ""),
		register.NewConst("ErrBadHTML", template.ErrBadHTML, ""),
		register.NewConst("ErrBranchEnd", template.ErrBranchEnd, ""),
		register.NewConst("ErrEndContext", template.ErrEndContext, ""),
		register.NewConst("ErrNoSuchTemplate", template.ErrNoSuchTemplate, ""),
		register.NewConst("ErrOutputContext", template.ErrOutputContext, ""),
		register.NewConst("ErrPartialCharset", template.ErrPartialCharset, ""),
		register.NewConst("ErrPartialEscape", template.ErrPartialEscape, ""),
		register.NewConst("ErrPredefinedEscaper", template.ErrPredefinedEscaper, ""),
		register.NewConst("ErrRangeLoopReentry", template.ErrRangeLoopReentry, ""),
		register.NewConst("ErrSlashAmbig", template.ErrSlashAmbig, ""),
		register.NewType("Error", reflect.TypeOf(func(template.Error) {}).In(0), ""),
		register.NewType("ErrorCode", reflect.TypeOf(func(template.ErrorCode) {}).In(0), ""),
		register.NewType("FuncMap", reflect.TypeOf(func(template.FuncMap) {}).In(0), ""),
		register.NewType("HTML", reflect.TypeOf(func(template.HTML) {}).In(0), ""),
		register.NewType("HTMLAttr", reflect.TypeOf(func(template.HTMLAttr) {}).In(0), ""),
		register.NewFunction("HTMLEscape", template.HTMLEscape, ""),
		register.NewFunction("HTMLEscapeString", template.HTMLEscapeString, ""),
		register.NewFunction("HTMLEscaper", template.HTMLEscaper, ""),
		register.NewFunction("IsTrue", template.IsTrue, ""),
		register.NewType("JS", reflect.TypeOf(func(template.JS) {}).In(0), ""),
		register.NewFunction("JSEscape", template.JSEscape, ""),
		register.NewFunction("JSEscapeString", template.JSEscapeString, ""),
		register.NewFunction("JSEscaper", template.JSEscaper, ""),
		register.NewType("JSStr", reflect.TypeOf(func(template.JSStr) {}).In(0), ""),
		register.NewFunction("Must", template.Must, ""),
		register.NewFunction("New", template.New, ""),
		register.NewConst("OK", template.OK, ""),
		register.NewFunction("ParseFiles", template.ParseFiles, ""),
		register.NewFunction("ParseGlob", template.ParseGlob, ""),
		register.NewType("Srcset", reflect.TypeOf(func(template.Srcset) {}).In(0), ""),
		register.NewType("Template", reflect.TypeOf(func(template.Template) {}).In(0), ""),
		register.NewType("URL", reflect.TypeOf(func(template.URL) {}).In(0), ""),
		register.NewFunction("URLQueryEscaper", template.URLQueryEscaper, ""),
	)
}
