package imports

import (
	"reflect"
	"strconv"

	"git.woa.com/vasd_masc_ba/YitihuaOteam/base/gofun/register"
)

var _ = reflect.Int

func init() {
	register.AddPackage("strconv", "strconv",
		register.NewFunction("AppendBool", strconv.AppendBool, ""),
		register.NewFunction("AppendFloat", strconv.AppendFloat, ""),
		register.NewFunction("AppendInt", strconv.AppendInt, ""),
		register.NewFunction("AppendQuote", strconv.AppendQuote, ""),
		register.NewFunction("AppendQuoteRune", strconv.AppendQuoteRune, ""),
		register.NewFunction("AppendQuoteRuneToASCII", strconv.AppendQuoteRuneToASCII, ""),
		register.NewFunction("AppendQuoteRuneToGraphic", strconv.AppendQuoteRuneToGraphic, ""),
		register.NewFunction("AppendQuoteToASCII", strconv.AppendQuoteToASCII, ""),
		register.NewFunction("AppendQuoteToGraphic", strconv.AppendQuoteToGraphic, ""),
		register.NewFunction("AppendUint", strconv.AppendUint, ""),
		register.NewFunction("Atoi", strconv.Atoi, ""),
		register.NewFunction("CanBackquote", strconv.CanBackquote, ""),
		register.NewVar("ErrRange", &strconv.ErrRange, reflect.TypeOf(func(error) {}).In(0), ""),
		register.NewVar("ErrSyntax", &strconv.ErrSyntax, reflect.TypeOf(func(error) {}).In(0), ""),
		register.NewFunction("FormatBool", strconv.FormatBool, ""),
		register.NewFunction("FormatFloat", strconv.FormatFloat, ""),
		register.NewFunction("FormatInt", strconv.FormatInt, ""),
		register.NewFunction("FormatUint", strconv.FormatUint, ""),
		register.NewConst("IntSize", strconv.IntSize, ""),
		register.NewFunction("IsGraphic", strconv.IsGraphic, ""),
		register.NewFunction("IsPrint", strconv.IsPrint, ""),
		register.NewFunction("Itoa", strconv.Itoa, ""),
		register.NewType("NumError", reflect.TypeOf(func(strconv.NumError) {}).In(0), ""),
		register.NewFunction("ParseBool", strconv.ParseBool, ""),
		register.NewFunction("ParseFloat", strconv.ParseFloat, ""),
		register.NewFunction("ParseInt", strconv.ParseInt, ""),
		register.NewFunction("ParseUint", strconv.ParseUint, ""),
		register.NewFunction("Quote", strconv.Quote, ""),
		register.NewFunction("QuoteRune", strconv.QuoteRune, ""),
		register.NewFunction("QuoteRuneToASCII", strconv.QuoteRuneToASCII, ""),
		register.NewFunction("QuoteRuneToGraphic", strconv.QuoteRuneToGraphic, ""),
		register.NewFunction("QuoteToASCII", strconv.QuoteToASCII, ""),
		register.NewFunction("QuoteToGraphic", strconv.QuoteToGraphic, ""),
		register.NewFunction("Unquote", strconv.Unquote, ""),
		register.NewFunction("UnquoteChar", strconv.UnquoteChar, ""),
	)
}
