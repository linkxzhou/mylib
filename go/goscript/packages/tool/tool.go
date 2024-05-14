package main

import (
	"fmt"
	"go/ast"
	"go/format"
	"go/importer"
	"go/token"
	"go/types"
	"io/ioutil"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"git.woa.com/vasd_masc_ba/YitihuaOteam/base/gofun/packages/tool/pkgs"
)

var sourceDir string

func init() {
	_, filename, _, _ := runtime.Caller(0)
	sourceDir = filepath.Dir(filename)
}

//go:generate go run tool.go
func main() {
	for path, vlist := range pkgs.ImportPkgs {
		fmt.Println("path: ", path, ", vlist: ", vlist)
		err := packageImport(path, vlist)
		if err != nil {
			println(path, err.Error())
			continue
		}
	}
}

// objectDecl 根据object的类型，生成相应的注册声明
func objectDecl(object types.Object) string {
	name := fmt.Sprintf("%s.%s", object.Pkg().Name(), object.Name())

	switch object.(type) {
	case *types.TypeName:
		return fmt.Sprintf(`register.NewType("%s", reflect.TypeOf(func(%s){}).In(0), "%s")`, object.Name(), name, "")
	case *types.Const:
		if object.Name() == "MaxUint64" {
			// 用整型常量传参时会被自动转换成int，MaxUint64需要特殊处理转换为uint，否则会编译报错
			name = fmt.Sprintf("uint(%s)", name)
		}
		return fmt.Sprintf(`register.NewConst("%s", %s, "%s")`, object.Name(), name, "")
	case *types.Var:
		switch object.Type().Underlying().(type) {
		case *types.Interface:
			return fmt.Sprintf(`register.NewVar("%s", &%s, reflect.TypeOf(func (%s){}).In(0), "%s")`, object.Name(), name, trimVendor(object.Type().String()), "")
		default:
			return fmt.Sprintf(`register.NewVar("%s", &%s, reflect.TypeOf(%s), "%s")`, object.Name(), name, name, "")
		}

	case *types.Func:
		return fmt.Sprintf(`register.NewFunction("%s", %s, "%s")`, object.Name(), name, "")
	}
	return ""
}

func trimVendor(src string) string {
	if i := strings.LastIndex(src, `vendor/`); i >= 0 {
		return src[i+7:]
	}
	return src
}

func packageImport(path string, vlist []string) error {
	pkg, err := importer.ForCompiler(token.NewFileSet(), "source", nil).Import(path)
	if err != nil {
		return err
	}

	preImports := ""
	for _, v := range vlist {
		preImports = preImports + `"` + v + `"` + "\n"
	}

	builder := strings.Builder{}
	pkgPath := trimVendor(pkg.Path())
	fmt.Println("pkg.Path(): ", pkg.Path(), ", pkgPath: ", pkgPath)
	builder.WriteString(fmt.Sprintf(`package imports
import (
	%s
	"%s"
	"reflect"

	"git.woa.com/vasd_masc_ba/YitihuaOteam/base/gofun/register"
)

var _ = reflect.Int

func init() {
	register.AddPackage("%s", "%s",`+"\n", preImports, path, path, pkg.Name()))
	scope := pkg.Scope()
	for _, declName := range pkg.Scope().Names() {
		if ast.IsExported(declName) {
			// 为所有可导出的对象生成注册语句
			obj := scope.Lookup(declName)
			builder.WriteString(strings.Replace(objectDecl(obj), path, pkg.Name(), 1) + ",\n")
		}
	}
	builder.WriteString(`)
}`)

	src := builder.String()
	code, err := format.Source([]byte(src))
	if err != nil {
		code = []byte(src)
		println(path, err.Error())
	}
	fmt.Println("pkg.Name(): ", pkg.Name())
	filename := fmt.Sprintf("%s%c%s.go", filepath.Dir(sourceDir), os.PathSeparator, pkg.Name())
	return ioutil.WriteFile(filename, code, 0666)
}
