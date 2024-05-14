package imports

import (
	"encoding/binary"
	"reflect"

	"git.woa.com/vasd_masc_ba/YitihuaOteam/base/gofun/register"
)

var _ = reflect.Int

func init() {
	register.AddPackage("encoding/binary", "binary",
		register.NewVar("BigEndian", &binary.BigEndian, reflect.TypeOf(binary.BigEndian), ""),
		register.NewType("ByteOrder", reflect.TypeOf(func(binary.ByteOrder) {}).In(0), ""),
		register.NewVar("LittleEndian", &binary.LittleEndian, reflect.TypeOf(binary.LittleEndian), ""),
		register.NewConst("MaxVarintLen16", binary.MaxVarintLen16, ""),
		register.NewConst("MaxVarintLen32", binary.MaxVarintLen32, ""),
		register.NewConst("MaxVarintLen64", binary.MaxVarintLen64, ""),
		register.NewFunction("PutUvarint", binary.PutUvarint, ""),
		register.NewFunction("PutVarint", binary.PutVarint, ""),
		register.NewFunction("Read", binary.Read, ""),
		register.NewFunction("ReadUvarint", binary.ReadUvarint, ""),
		register.NewFunction("ReadVarint", binary.ReadVarint, ""),
		register.NewFunction("Size", binary.Size, ""),
		register.NewFunction("Uvarint", binary.Uvarint, ""),
		register.NewFunction("Varint", binary.Varint, ""),
		register.NewFunction("Write", binary.Write, ""),
	)
}
