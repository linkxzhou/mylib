package imports

import (
	"reflect"
	"sync"

	"git.woa.com/vasd_masc_ba/YitihuaOteam/base/gofun/register"
)

var _ = reflect.Int

func init() {
	register.AddPackage("sync", "sync",
		register.NewType("Cond", reflect.TypeOf(func(sync.Cond) {}).In(0), ""),
		register.NewType("Locker", reflect.TypeOf(func(sync.Locker) {}).In(0), ""),
		register.NewType("Map", reflect.TypeOf(func(sync.Map) {}).In(0), ""),
		register.NewType("Mutex", reflect.TypeOf(func(sync.Mutex) {}).In(0), ""),
		register.NewFunction("NewCond", sync.NewCond, ""),
		register.NewType("Once", reflect.TypeOf(func(sync.Once) {}).In(0), ""),
		register.NewType("Pool", reflect.TypeOf(func(sync.Pool) {}).In(0), ""),
		register.NewType("RWMutex", reflect.TypeOf(func(sync.RWMutex) {}).In(0), ""),
		register.NewType("WaitGroup", reflect.TypeOf(func(sync.WaitGroup) {}).In(0), ""),
	)
}
