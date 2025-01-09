package imports

import (
	"fmt"
	"reflect"

	"git.woa.com/vasd_masc_ba/YitihuaOteam/base/gofun/register"
)

var _ = reflect.Int

func init() {
	register.AddPackage("fmt", "fmt",
		register.NewFunction("Errorf", fmt.Errorf, ""),
		register.NewType("Formatter", reflect.TypeOf(func(fmt.Formatter) {}).In(0), ""),
		register.NewFunction("Fprint", fmt.Fprint, ""),
		register.NewFunction("Fprintf", fmt.Fprintf, ""),
		register.NewFunction("Fprintln", fmt.Fprintln, ""),
		register.NewFunction("Fscan", fmt.Fscan, ""),
		register.NewFunction("Fscanf", fmt.Fscanf, ""),
		register.NewFunction("Fscanln", fmt.Fscanln, ""),
		register.NewType("GoStringer", reflect.TypeOf(func(fmt.GoStringer) {}).In(0), ""),
		register.NewFunction("Print", fmt.Print, ""),
		register.NewFunction("Printf", fmt.Printf, ""),
		register.NewFunction("Println", fmt.Println, ""),
		register.NewFunction("Scan", fmt.Scan, ""),
		register.NewType("ScanState", reflect.TypeOf(func(fmt.ScanState) {}).In(0), ""),
		register.NewFunction("Scanf", fmt.Scanf, ""),
		register.NewFunction("Scanln", fmt.Scanln, ""),
		register.NewType("Scanner", reflect.TypeOf(func(fmt.Scanner) {}).In(0), ""),
		register.NewFunction("Sprint", fmt.Sprint, ""),
		register.NewFunction("Sprintf", fmt.Sprintf, ""),
		register.NewFunction("Sprintln", fmt.Sprintln, ""),
		register.NewFunction("Sscan", fmt.Sscan, ""),
		register.NewFunction("Sscanf", fmt.Sscanf, ""),
		register.NewFunction("Sscanln", fmt.Sscanln, ""),
		register.NewType("State", reflect.TypeOf(func(fmt.State) {}).In(0), ""),
		register.NewType("Stringer", reflect.TypeOf(func(fmt.Stringer) {}).In(0), ""),
	)
}
