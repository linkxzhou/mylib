package imports

import (
	"io"
	"reflect"

	"git.woa.com/vasd_masc_ba/YitihuaOteam/base/gofun/register"
)

var _ = reflect.Int

func init() {
	register.AddPackage("io", "io",
		register.NewType("ByteReader", reflect.TypeOf(func(io.ByteReader) {}).In(0), ""),
		register.NewType("ByteScanner", reflect.TypeOf(func(io.ByteScanner) {}).In(0), ""),
		register.NewType("ByteWriter", reflect.TypeOf(func(io.ByteWriter) {}).In(0), ""),
		register.NewType("Closer", reflect.TypeOf(func(io.Closer) {}).In(0), ""),
		register.NewFunction("Copy", io.Copy, ""),
		register.NewFunction("CopyBuffer", io.CopyBuffer, ""),
		register.NewFunction("CopyN", io.CopyN, ""),
		register.NewVar("EOF", &io.EOF, reflect.TypeOf(func(error) {}).In(0), ""),
		register.NewVar("ErrClosedPipe", &io.ErrClosedPipe, reflect.TypeOf(func(error) {}).In(0), ""),
		register.NewVar("ErrNoProgress", &io.ErrNoProgress, reflect.TypeOf(func(error) {}).In(0), ""),
		register.NewVar("ErrShortBuffer", &io.ErrShortBuffer, reflect.TypeOf(func(error) {}).In(0), ""),
		register.NewVar("ErrShortWrite", &io.ErrShortWrite, reflect.TypeOf(func(error) {}).In(0), ""),
		register.NewVar("ErrUnexpectedEOF", &io.ErrUnexpectedEOF, reflect.TypeOf(func(error) {}).In(0), ""),
		register.NewFunction("LimitReader", io.LimitReader, ""),
		register.NewType("LimitedReader", reflect.TypeOf(func(io.LimitedReader) {}).In(0), ""),
		register.NewFunction("MultiReader", io.MultiReader, ""),
		register.NewFunction("MultiWriter", io.MultiWriter, ""),
		register.NewFunction("NewSectionReader", io.NewSectionReader, ""),
		register.NewFunction("Pipe", io.Pipe, ""),
		register.NewType("PipeReader", reflect.TypeOf(func(io.PipeReader) {}).In(0), ""),
		register.NewType("PipeWriter", reflect.TypeOf(func(io.PipeWriter) {}).In(0), ""),
		register.NewFunction("ReadAtLeast", io.ReadAtLeast, ""),
		register.NewType("ReadCloser", reflect.TypeOf(func(io.ReadCloser) {}).In(0), ""),
		register.NewFunction("ReadFull", io.ReadFull, ""),
		register.NewType("ReadSeeker", reflect.TypeOf(func(io.ReadSeeker) {}).In(0), ""),
		register.NewType("ReadWriteCloser", reflect.TypeOf(func(io.ReadWriteCloser) {}).In(0), ""),
		register.NewType("ReadWriteSeeker", reflect.TypeOf(func(io.ReadWriteSeeker) {}).In(0), ""),
		register.NewType("ReadWriter", reflect.TypeOf(func(io.ReadWriter) {}).In(0), ""),
		register.NewType("Reader", reflect.TypeOf(func(io.Reader) {}).In(0), ""),
		register.NewType("ReaderAt", reflect.TypeOf(func(io.ReaderAt) {}).In(0), ""),
		register.NewType("ReaderFrom", reflect.TypeOf(func(io.ReaderFrom) {}).In(0), ""),
		register.NewType("RuneReader", reflect.TypeOf(func(io.RuneReader) {}).In(0), ""),
		register.NewType("RuneScanner", reflect.TypeOf(func(io.RuneScanner) {}).In(0), ""),
		register.NewType("SectionReader", reflect.TypeOf(func(io.SectionReader) {}).In(0), ""),
		register.NewConst("SeekCurrent", io.SeekCurrent, ""),
		register.NewConst("SeekEnd", io.SeekEnd, ""),
		register.NewConst("SeekStart", io.SeekStart, ""),
		register.NewType("Seeker", reflect.TypeOf(func(io.Seeker) {}).In(0), ""),
		register.NewType("StringWriter", reflect.TypeOf(func(io.StringWriter) {}).In(0), ""),
		register.NewFunction("TeeReader", io.TeeReader, ""),
		register.NewType("WriteCloser", reflect.TypeOf(func(io.WriteCloser) {}).In(0), ""),
		register.NewType("WriteSeeker", reflect.TypeOf(func(io.WriteSeeker) {}).In(0), ""),
		register.NewFunction("WriteString", io.WriteString, ""),
		register.NewType("Writer", reflect.TypeOf(func(io.Writer) {}).In(0), ""),
		register.NewType("WriterAt", reflect.TypeOf(func(io.WriterAt) {}).In(0), ""),
		register.NewType("WriterTo", reflect.TypeOf(func(io.WriterTo) {}).In(0), ""),
	)
}
