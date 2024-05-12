## Benchmark
expr = `((2 << 3) + (10 % 3)) * (5 - (x * 2)) + (3.0 / y) * (2.0 + 1.0) && ((z + "World") == "Hello World")`
args = `"x": 3.0, "y": 2.0, "z": "Hello "`

### no-cache
```
% go test -bench . -benchtime 10s -cpu 1,2,4,8,16 -benchmem
goos: darwin
goarch: arm64
pkg: github.com/linkxzhou/mylib/go/expr
BenchmarkExpr1       	11617224	       997.3 ns/op	     400 B/op	      34 allocs/op
BenchmarkExpr1-2     	13021327	       917.5 ns/op	     400 B/op	      34 allocs/op
BenchmarkExpr1-4     	12983631	       918.2 ns/op	     400 B/op	      34 allocs/op
BenchmarkExpr1-8     	13066659	       930.7 ns/op	     400 B/op	      34 allocs/op
BenchmarkExpr1-16    	12864978	       944.1 ns/op	     400 B/op	      34 allocs/op
PASS
ok  	github.com/linkxzhou/mylib/go/expr	64.852s
```

### cache
```
go test -bench . -benchtime 10s -cpu 1,2,4,8,16 -benchmem
goos: darwin
goarch: arm64
pkg: github.com/linkxzhou/mylib/go/expr
BenchmarkExpr1       	14427842	       820.0 ns/op	     304 B/op	      27 allocs/op
BenchmarkExpr1-2     	15549236	       757.6 ns/op	     304 B/op	      27 allocs/op
BenchmarkExpr1-4     	15747040	       755.1 ns/op	     304 B/op	      27 allocs/op
BenchmarkExpr1-8     	15788304	       760.5 ns/op	     304 B/op	      27 allocs/op
BenchmarkExpr1-16    	15724354	       760.1 ns/op	     304 B/op	      27 allocs/op
PASS
ok  	github.com/linkxzhou/mylib/go/expr	63.644s
```

### new version
```
lvzhou@lvdeMacBook-Pro expr % go test -bench . -benchtime 1s -cpu 4 -benchmem -cpuprofile cpu.pprof -memprofile mem.pprof
goos: darwin
goarch: arm64
pkg: github.com/linkxzhou/mylib/go/expr
BenchmarkExprNoCache-4   	 1653823	       704.2 ns/op	     248 B/op	      24 allocs/op
BenchmarkExprCache-4     	 2193422	       543.7 ns/op	     192 B/op	      19 allocs/op
BenchmarkSystem-4        	100000000	        11.24 ns/op	       0 B/op	       0 allocs/op
PASS
ok  	github.com/linkxzhou/mylib/go/expr	5.394s
````