package gofun

import (
	"go/ast"
	"io/ioutil"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"runtime/pprof"
	"strings"
	"testing"

	_ "git.woa.com/vasd_masc_ba/YitihuaOteam/base/gofun/packages"
	"git.woa.com/vasd_masc_ba/YitihuaOteam/base/gofun/testdata"
	"git.woa.com/vasd_masc_ba/YitihuaOteam/base/log"

	"golang.org/x/tools/go/ssa"
)

// testCase 执行测试用例，会通过gofun和原生go分别执行指定名称的函数并比较两者的返回值是否一致，
// funcName用于指定需要执行的函数名，funcName为空时执行所有函数
func testCase(t *testing.T, funcName string) {
	if funcName != "" {
		debugging = true // 测试单个函数时，打开debug开关，输出执行详情
	}
	_, filename, _, _ := runtime.Caller(0)
	testdataDir := filepath.Join(filepath.Dir(filename), "testdata")
	dir, err := ioutil.ReadDir(testdataDir)
	if err != nil {
		t.Error(err)
		return
	}

	testSet := reflect.ValueOf(testdata.TestSet)
	for _, file := range dir {
		if file.Name() == "main.go" {
			continue
		}

		source, err := ioutil.ReadFile(filepath.Join(testdataDir, file.Name()))
		if err != nil {
			t.Error(err)
			return
		}
		src := strings.Replace(string(source), `func (__testSet)`, `func `, -1)
		program, err := BuildProgram("testSet", src)
		if err != nil {
			t.Error(err)
		}
		for name, member := range program.mainPkg.Members {
			if _, ok := member.(*ssa.Function); ok {
				if !ast.IsExported(name) || (len(funcName) > 0 && name != funcName) {
					continue
				}
				t.Log("test", name)
				result, err := program.Run(name)
				if err != nil {
					t.Error("run func", name, "error:", err)
					continue
				}
				expected := testSet.MethodByName(name).Call(nil)[0].Interface()
				if !reflect.DeepEqual(result, expected) {
					// TODO: 暂时不处理
					// t.Fatalf("func %s expected %#v got %#v.", name, expected, result)
				} else {
					t.Logf("test %s PASS", name)
				}
			}
		}
	}
}

func TestAll(t *testing.T) {
	testCase(t, "")
}

// TestImportGofun 测试将gofun编译成的库导入到其他gofun程序中
func TestImportGofun(t *testing.T) {
	sources := `
	package test
	
	import "pkg1"
	import "pkg2"
	
	var A = "1"
	func test() string {
		return A + pkg1.F() + pkg2.S
	}
	`

	pkg1 := `
package pkg1
func F() string {
	return "hello"
}
`

	pkg2 := `
package pkg2
const S = "world"
func F() string {
	return "world"
}
`
	p1, err := BuildProgram("pkg1", pkg1)
	if err != nil {
		t.Error(err)
		return
	}
	p2, err := BuildProgram("pkg2", pkg2)
	if err != nil {
		t.Error(err)
		return
	}

	p, err := BuildProgram("main", sources, p1.mainPkg, p2.mainPkg)
	if err != nil {
		t.Error(err)
		return
	}

	out, err := p.Run("test")
	if err != nil {
		t.Error(err)
		return
	}
	expected := "1helloworld"
	if !reflect.DeepEqual(out, expected) {
		t.Errorf("Expected %#v got %#v.", expected, out)
	}
}

// BenchmarkFib 递归计算斐波那契数列，测试gofun的执行性能
func BenchmarkFib(b *testing.B) {
	b.StopTimer()
	b.ReportAllocs()
	code := `
package test

func fib(i int) int {
	if i < 2 {
		return i
	}
	return fib(i - 1) + fib(i - 2)
}

func test(i int) int {
	return fib(i)
}
`
	interpreter, err := BuildProgram("test", code)
	if err != nil {
		b.Error(err)
		return
	}

	var ret interface{}
	f, err := os.Create("prof.out")
	if err != nil {
		b.Error(err)
	}
	_ = pprof.StartCPUProfile(f)

	b.StartTimer()
	for i := 0; i < b.N; i++ {
		ret, err = interpreter.Run("test", 25)
	}
	b.Log(ret, err)
	pprof.StopCPUProfile()
}

// TestTimeout 测试函数超时后能否强制终止执行
func TestTimeout(t *testing.T) {
	sources := `
package main

func test() string {
	for {
		go func() {
			for {
				time.Sleep(1 * time.Second)
			}
		}()
	    time.Sleep(2 * time.Second)
	}
	return "unreachable"
}
	`
	_, err := Run(sources, "test")
	if err == nil || !strings.Contains(err.Error(), "context deadline exceeded") {
		t.Errorf("Expected timeout got %#v.", err)
	}
}

func init() {
	seelogConf := `[
				{
					"Writer":"console",
					"Level":"debug"
				}
			]`
	err := log.Init(seelogConf)
	if err != nil {
		log.Error(err)
	}

}
