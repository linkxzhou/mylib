package imports

import (
	"reflect"
	"sync/atomic"

	"git.woa.com/vasd_masc_ba/YitihuaOteam/base/gofun/register"
)

var _ = reflect.Int

func init() {
	register.AddPackage("sync/atomic", "atomic",
		register.NewFunction("AddInt32", atomic.AddInt32, ""),
		register.NewFunction("AddInt64", atomic.AddInt64, ""),
		register.NewFunction("AddUint32", atomic.AddUint32, ""),
		register.NewFunction("AddUint64", atomic.AddUint64, ""),
		register.NewFunction("AddUintptr", atomic.AddUintptr, ""),
		register.NewFunction("CompareAndSwapInt32", atomic.CompareAndSwapInt32, ""),
		register.NewFunction("CompareAndSwapInt64", atomic.CompareAndSwapInt64, ""),
		register.NewFunction("CompareAndSwapPointer", atomic.CompareAndSwapPointer, ""),
		register.NewFunction("CompareAndSwapUint32", atomic.CompareAndSwapUint32, ""),
		register.NewFunction("CompareAndSwapUint64", atomic.CompareAndSwapUint64, ""),
		register.NewFunction("CompareAndSwapUintptr", atomic.CompareAndSwapUintptr, ""),
		register.NewFunction("LoadInt32", atomic.LoadInt32, ""),
		register.NewFunction("LoadInt64", atomic.LoadInt64, ""),
		register.NewFunction("LoadPointer", atomic.LoadPointer, ""),
		register.NewFunction("LoadUint32", atomic.LoadUint32, ""),
		register.NewFunction("LoadUint64", atomic.LoadUint64, ""),
		register.NewFunction("LoadUintptr", atomic.LoadUintptr, ""),
		register.NewFunction("StoreInt32", atomic.StoreInt32, ""),
		register.NewFunction("StoreInt64", atomic.StoreInt64, ""),
		register.NewFunction("StorePointer", atomic.StorePointer, ""),
		register.NewFunction("StoreUint32", atomic.StoreUint32, ""),
		register.NewFunction("StoreUint64", atomic.StoreUint64, ""),
		register.NewFunction("StoreUintptr", atomic.StoreUintptr, ""),
		register.NewFunction("SwapInt32", atomic.SwapInt32, ""),
		register.NewFunction("SwapInt64", atomic.SwapInt64, ""),
		register.NewFunction("SwapPointer", atomic.SwapPointer, ""),
		register.NewFunction("SwapUint32", atomic.SwapUint32, ""),
		register.NewFunction("SwapUint64", atomic.SwapUint64, ""),
		register.NewFunction("SwapUintptr", atomic.SwapUintptr, ""),
		register.NewType("Value", reflect.TypeOf(func(atomic.Value) {}).In(0), ""),
	)
}
