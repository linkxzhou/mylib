package imports

import (
	"crypto/cipher"
	"reflect"

	"git.woa.com/vasd_masc_ba/YitihuaOteam/base/gofun/register"
)

var _ = reflect.Int

func init() {
	register.AddPackage("crypto/cipher", "cipher",
		register.NewType("AEAD", reflect.TypeOf(func(cipher.AEAD) {}).In(0), ""),
		register.NewType("Block", reflect.TypeOf(func(cipher.Block) {}).In(0), ""),
		register.NewType("BlockMode", reflect.TypeOf(func(cipher.BlockMode) {}).In(0), ""),
		register.NewFunction("NewCBCDecrypter", cipher.NewCBCDecrypter, ""),
		register.NewFunction("NewCBCEncrypter", cipher.NewCBCEncrypter, ""),
		register.NewFunction("NewCFBDecrypter", cipher.NewCFBDecrypter, ""),
		register.NewFunction("NewCFBEncrypter", cipher.NewCFBEncrypter, ""),
		register.NewFunction("NewCTR", cipher.NewCTR, ""),
		register.NewFunction("NewGCM", cipher.NewGCM, ""),
		register.NewFunction("NewGCMWithNonceSize", cipher.NewGCMWithNonceSize, ""),
		register.NewFunction("NewGCMWithTagSize", cipher.NewGCMWithTagSize, ""),
		register.NewFunction("NewOFB", cipher.NewOFB, ""),
		register.NewType("Stream", reflect.TypeOf(func(cipher.Stream) {}).In(0), ""),
		register.NewType("StreamReader", reflect.TypeOf(func(cipher.StreamReader) {}).In(0), ""),
		register.NewType("StreamWriter", reflect.TypeOf(func(cipher.StreamWriter) {}).In(0), ""),
	)
}
