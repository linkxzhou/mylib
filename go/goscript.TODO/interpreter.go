package gofun

import (
	"errors"
	"fmt"
	"go/ast"
	"go/doc"
	"go/parser"
	"go/token"
	"go/types"
	"os"

	// goimporter "go/importer"

	"git.woa.com/vasd_masc_ba/YitihuaOteam/base/gofun/internal/importer"
	"git.woa.com/vasd_masc_ba/YitihuaOteam/base/gofun/internal/value"
	"git.woa.com/vasd_masc_ba/YitihuaOteam/base/log"

	"golang.org/x/tools/go/ssa"
	"golang.org/x/tools/go/ssa/ssautil"
)

// Program 代码编译后生成的结构体
type Program struct {
	mainPkg   *ssa.Package
	globals   map[ssa.Value]*value.Value // addresses of global variables (immutable)
	importPkg []string                   // 导入的包信息
}

// 获取函数列表信息，exportedAll判断是否导出所有的函数
func ParseFuncList(sourceCode string, exportedAll bool) ([]string, error) {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "main.go", sourceCode, parser.AllErrors)
	if err != nil {
		log.Error("ParseFile source: ", sourceCode, " failed")
		return nil, err
	}
	var flist []string
	for _, v := range f.Decls {
		switch t := v.(type) {
		case *ast.FuncDecl:
			// f.Recv != nil 标识存在struct的成员函数
			if f, ok := v.(*ast.FuncDecl); !ok || f.Recv != nil {
				continue
			}
			if exportedAll {
				flist = append(flist, t.Name.Name)
			} else {
				if t.Name.IsExported() && t.Name.Name != "init" {
					flist = append(flist, t.Name.Name)
				}
			}
		default:
			// pass
		}
	}
	return flist, nil
}

// Run 解释执行代码中的函数
func Run(sourceCode string, funcName string, params ...interface{}) (interface{}, error) {
	program, err := BuildProgram("main", sourceCode)
	if err != nil {
		return nil, err
	}
	return program.Run(funcName, params...)
}

// autoImport 自动添加import语句
func autoImport(f *ast.File) []string {
	var importPkg []string
	imported := make(map[string]bool)
	for _, i := range f.Imports {
		imported[i.Path.Value] = true
		importPkg = append(importPkg, i.Path.Value)
	}
	for _, unresolved := range f.Unresolved {
		if doc.IsPredeclared(unresolved.Name) {
			continue
		}
		if importSpec := importer.GetPackageByName(unresolved.Name); importSpec != nil {
			if imported[importSpec.Path.Value] {
				continue
			}
			imported[importSpec.Path.Value] = true
			f.Imports = append(f.Imports, importSpec)
			f.Decls = append(f.Decls, &ast.GenDecl{
				Specs: []ast.Spec{importSpec},
			})
		}
	}
	return importPkg
}

// BuildProgram 编译代码
func BuildProgram(fname, sourceCode string, packages ...*ssa.Package) (*Program, error) {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, fname+".go", sourceCode, parser.AllErrors)
	if err != nil {
		return nil, err
	}
	files := []*ast.File{f}

	importPkg := autoImport(f)
	pkg := types.NewPackage(f.Name.Name, f.Name.Name)

	packageImporter := importer.NewImporter(packages...)
	mode := ssa.SanityCheckFunctions | ssa.BareInits
	mainPkg, _, err := ssautil.BuildPackage(
		&types.Config{Importer: packageImporter}, fset, pkg, files, mode)
	if err != nil {
		return nil, err
	}
	program := &Program{
		mainPkg:   mainPkg,
		globals:   make(map[ssa.Value]*value.Value),
		importPkg: importPkg, // 导入的包列表信息
	}
	value.ExternalValueWrap(packageImporter, mainPkg)
	program.initGlobal()
	context := newCallContext()
	fr := &frame{program: program, context: context}
	if init := mainPkg.Func("init"); init != nil {
		for _, pkg := range packages {
			if dependInit := pkg.Func("init"); dependInit != nil {
				callSSA(fr, dependInit, nil, nil)
			}
		}
		// init初始化函数
		callSSA(fr, init, nil, nil)
	}
	context.cancelFunc()
	return program, nil
}

// Run 执行函数
func (p *Program) Run(funcName string, params ...interface{}) (interface{}, error) {
	val, _, err := p.RunWithContext(funcName, params...)
	return val, err
}

// RunWithContext 执行函数并返回上下文
func (p *Program) RunWithContext(funcName string, params ...interface{}) (result interface{}, ctx *Context, err error) {
	defer func() {
		if re := recover(); re != nil {
			err = fmt.Errorf("%v", re)
		}
	}()
	ctx = newCallContext()
	mainFn := p.mainPkg.Func(funcName)
	if mainFn == nil {
		return nil, nil, errors.New("function not found")
	}
	if debugging {
		_, _ = mainFn.WriteTo(os.Stdout)
	}
	args := make([]value.Value, len(params))
	for i := range args {
		args[i] = value.ValueOf(params[i])
	}
	fr := &frame{
		program: p,
		context: ctx,
	}
	ret := callSSA(fr, mainFn, args, nil)
	if ret != nil {
		result = ret.Interface()
	}
	ctx.cancelFunc()
	return
}

func (p *Program) initGlobal() {
	for _, v := range p.mainPkg.Members {
		if g, ok := v.(*ssa.Global); ok {
			global := zero(g.Type().(*types.Pointer).Elem()).Elem()
			p.globals[g] = &global
		}
	}
}

// SetGlobalValue 修改全局变量的值
func (p *Program) SetGlobalValue(name string, val interface{}) error {
	v := p.mainPkg.Members[name]
	if g, ok := v.(*ssa.Global); ok {
		global := value.ValueOf(val)
		p.globals[g] = &global
		return nil
	}
	return fmt.Errorf("global Value %s not found", name)
}

// Package 获取package，用于导入到其他package
func (p *Program) Package() *ssa.Package {
	return p.mainPkg
}
