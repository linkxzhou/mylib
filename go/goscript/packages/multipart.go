package imports

import (
	"mime/multipart"
	"reflect"

	"git.woa.com/vasd_masc_ba/YitihuaOteam/base/gofun/register"
)

var _ = reflect.Int

func init() {
	register.AddPackage("mime/multipart", "multipart",
		register.NewVar("ErrMessageTooLarge", &multipart.ErrMessageTooLarge, reflect.TypeOf(func(error) {}).In(0), ""),
		register.NewType("File", reflect.TypeOf(func(multipart.File) {}).In(0), ""),
		register.NewType("FileHeader", reflect.TypeOf(func(multipart.FileHeader) {}).In(0), ""),
		register.NewType("Form", reflect.TypeOf(func(multipart.Form) {}).In(0), ""),
		register.NewFunction("NewReader", multipart.NewReader, ""),
		register.NewFunction("NewWriter", multipart.NewWriter, ""),
		register.NewType("Part", reflect.TypeOf(func(multipart.Part) {}).In(0), ""),
		register.NewType("Reader", reflect.TypeOf(func(multipart.Reader) {}).In(0), ""),
		register.NewType("Writer", reflect.TypeOf(func(multipart.Writer) {}).In(0), ""),
	)
}
