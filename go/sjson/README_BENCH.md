# 性能对比

## 1. lexer.go性能优化
测试命令：go test -bench=BenchmarkLexer -benchmem -benchtime=10s -cpuprofile=cpu.prof -memprofile=mem.prof
```
第一轮：
BenchmarkLexerComplex-14        24878343          2867 ns/op         376 B/op         33 allocs/op

第二轮：
BenchmarkLexerComplex-14        35005912          2117 ns/op        1056 B/op         33 allocs/op

第三轮：
BenchmarkLexerComplex-14        51376759          1407 ns/op        1056 B/op         33 allocs/op

第四轮：
BenchmarkLexerSimple-14             	36590464	       315.0 ns/op	     224 B/op	       7 allocs/op
BenchmarkLexerComplex-14            	 8707606	      1386 ns/op	    1056 B/op	      33 allocs/op
BenchmarkLexerStringEscapes-14      	54042236	       219.0 ns/op	     128 B/op	       3 allocs/op
BenchmarkLexerAllTokens-14          	 7791068	      1515 ns/op	     832 B/op	      25 allocs/op
BenchmarkLexerBatchProcessing/单个标记处理-14                     	 7400288	      1519 ns/op	     832 B/op	      25 allocs/op
BenchmarkLexerBatchProcessing/批量标记处理-14                     	 4238590	      2892 ns/op	   10154 B/op	      33 allocs/op

第五轮：
BenchmarkLexerSimple-14             	42068858	       277.8 ns/op	     224 B/op	       7 allocs/op
BenchmarkLexerComplex-14            	 9167444	      1321 ns/op	    1056 B/op	      33 allocs/op
BenchmarkLexerStringEscapes-14      	53771250	       220.5 ns/op	     128 B/op	       3 allocs/op
BenchmarkLexerAllTokens-14          	 8859736	      1379 ns/op	     832 B/op	      25 allocs/op
BenchmarkLexerBatchProcessing/单个标记处理-14                     	 8427406	      1387 ns/op	     832 B/op	      25 allocs/op
BenchmarkLexerBatchProcessing/批量标记处理-14                     	 4499542	      2723 ns/op	   10154 B/op	      33 allocs/op

第六轮：
BenchmarkLexerSimple-14             	40208362	       279.9 ns/op	     224 B/op	       7 allocs/op
BenchmarkLexerComplex-14            	 9223904	      1339 ns/op	    1056 B/op	      33 allocs/op
BenchmarkLexerStringEscapes-14      	55456996	       217.9 ns/op	     128 B/op	       3 allocs/op
BenchmarkLexerAllTokens-14          	 8792521	      1359 ns/op	     832 B/op	      25 allocs/op
BenchmarkLexerBatchProcessing/单个标记处理-14                     	 8836092	      1373 ns/op	     832 B/op	      25 allocs/op
BenchmarkLexerBatchProcessing/批量标记处理-14                     	 4523031	      2782 ns/op	   10154 B/op	      33 allocs/op
```

## 2. parser.go性能优化
测试命令：go test -bench=BenchmarkParser -benchmem -benchtime=10s -cpuprofile=cpu.prof -memprofile=mem.prof
```
第一轮：
BenchmarkParserSimple-14              	41530178	       279.9 ns/op	     536 B/op	       9 allocs/op
BenchmarkParserComplex-14             	 5541475	      2175 ns/op	    3018 B/op	      62 allocs/op
BenchmarkParserArray-14               	19620277	       639.4 ns/op	    1128 B/op	      25 allocs/op
BenchmarkParserNestedStructures-14    	10833214	      1080 ns/op	    2802 B/op	      33 allocs/op
BenchmarkParserAllTypes-14            	 4809148	      2524 ns/op	    4908 B/op	      59 allocs/op
BenchmarkParserLargeDataset-14        	   84876	    140789 ns/op	  199811 B/op	    3905 allocs/op
BenchmarkParserDirectVsLexer/通过ParseJSON-14             	41691378	       292.7 ns/op	     536 B/op	       9 allocs/op
BenchmarkParserDirectVsLexer/手动Lexer+Parser-14          	42111043	       288.2 ns/op	     536 B/op	       9 allocs/op

第二轮：
BenchmarkParserSimple-14                27664126           437.8 ns/op      1442 B/op         11 allocs/op
BenchmarkParserComplex-14                4069184          2988 ns/op        7549 B/op         72 allocs/op
BenchmarkParserArray-14                 18862125           638.2 ns/op      1120 B/op         24 allocs/op
BenchmarkParserNestedStructures-14       5932318          2116 ns/op        8232 B/op         44 allocs/op
BenchmarkParserAllTypes-14               3982168          3038 ns/op        8817 B/op         65 allocs/op
BenchmarkParserLargeDataset-14             60127        199740 ns/op      471797 B/op       4505 allocs/op
BenchmarkParserDirectVsLexer/通过ParseJSON-14                 26527299           465.0 ns/op      1442 B/op         11 allocs/op
BenchmarkParserDirectVsLexer/手动Lexer+Parser-14              23275177           459.1 ns/op      1442 B/op         11 allocs/op

第三轮：
BenchmarkParserSimple-14              	36080635	       279.6 ns/op	     536 B/op	       9 allocs/op
BenchmarkParserComplex-14             	 5561377	      2160 ns/op	    3018 B/op	      62 allocs/op
BenchmarkParserArray-14               	19883038	       614.5 ns/op	    1120 B/op	      24 allocs/op
BenchmarkParserNestedStructures-14    	10681078	      1086 ns/op	    2794 B/op	      32 allocs/op
BenchmarkParserAllTypes-14            	 4826517	      2507 ns/op	    4892 B/op	      57 allocs/op
BenchmarkParserLargeDataset-14        	   86628	    139245 ns/op	  199802 B/op	    3904 allocs/op
BenchmarkParserDirectVsLexer/通过ParseJSON-14             	41779582	       289.6 ns/op	     536 B/op	       9 allocs/op
BenchmarkParserDirectVsLexer/手动Lexer+Parser-14          	41185029	       293.5 ns/op	     536 B/op	       9 allocs/op

第四轮：
BenchmarkParserSimple-14              	41273463	       282.0 ns/op	     536 B/op	       9 allocs/op
BenchmarkParserComplex-14             	 5553972	      2195 ns/op	    3018 B/op	      62 allocs/op
BenchmarkParserArray-14               	19403274	       627.5 ns/op	    1120 B/op	      24 allocs/op
BenchmarkParserNestedStructures-14    	11341714	      1084 ns/op	    2794 B/op	      32 allocs/op
BenchmarkParserAllTypes-14            	 4808508	      2563 ns/op	    4940 B/op	      58 allocs/op
BenchmarkParserLargeDataset-14        	   83746	    146739 ns/op	  199807 B/op	    3904 allocs/op
BenchmarkParserDirectVsLexer/通过ParseJSON-14             	41826134	       296.0 ns/op	     536 B/op	       9 allocs/op
BenchmarkParserDirectVsLexer/手动Lexer+Parser-14          	41891000	       290.0 ns/op	     536 B/op	       9 allocs/op
```

## 3. sjson.go性能优化
测试命令：go test -bench=BenchmarkCompareMedium -benchmem -benchtime=10s -cpuprofile=cpu.prof -memprofile=mem.prof
```
第一轮：
BenchmarkComplexJSON/Original-14         	13598248	      5525 ns/op	    9993 B/op	     178 allocs/op
BenchmarkComplexJSON/Optimized-14        	12703338	      5804 ns/op	   10645 B/op	     179 allocs/op
BenchmarkComplexJSON/Standard-14         	17148706	      4125 ns/op	    5136 B/op	     107 allocs/op

第二轮：
BenchmarkComplexDecode/SjsonDecode-14         	 3696825	      3378 ns/op	    5132 B/op	      89 allocs/op
BenchmarkComplexDecode/StdDecode-14           	 4004218	      3055 ns/op	    2632 B/op	      59 allocs/op

第三轮：
BenchmarkComplexDecode/SjsonDecode-14         	 7894840	      1569 ns/op	    2280 B/op	      40 allocs/op
BenchmarkComplexDecode/StdDecode-14           	 4115764	      3026 ns/op	    2632 B/op	      59 allocs/op

第四轮：
BenchmarkComplexEncode/SjsonEncode-14         	 2484872	      4986 ns/op	    5604 B/op	     153 allocs/op
BenchmarkComplexEncode/StdEncode-14           	 6173758	      2010 ns/op	    1841 B/op	      49 allocs/op

第五轮：优化 decode，直接解析，不经过 parser，代码见 sjson_direct_decode.go
BenchmarkDirectVsOldUnmarshal/OldParser-14         	 2506094	      4897 ns/op	    6838 B/op	     133 allocs/op
BenchmarkDirectVsOldUnmarshal/DirectParser-14      	 3106804	      3633 ns/op	    4341 B/op	     110 allocs/op
BenchmarkDirectVsOldUnmarshal/StdlibParser-14      	 2959168	      4138 ns/op	    3392 B/op	      81 allocs/op

第六轮：优化 encode，直接解析，不经过 parser，代码见 sjson_direct_encode.go
BenchmarkDirectComplexTypes/DirectMarshal-Complex-14         	 4028251	      3139 ns/op	    2139 B/op	      69 allocs/op
BenchmarkDirectComplexTypes/Marshal-Complex-14               	 3811060	      3081 ns/op	    2140 B/op	      69 allocs/op
BenchmarkDirectComplexTypes/StdMarshal-Complex-14            	11073609	      1111 ns/op	     880 B/op	      14 allocs/op

第七轮：
BenchmarkCompareMedium/SjsonEncode-14         	31065309	       356.5 ns/op	     472 B/op	       3 allocs/op
BenchmarkCompareMedium/StdEncode-14           	44274247	       244.7 ns/op	     216 B/op	       2 allocs/op
BenchmarkCompareMedium/JsoniterEncode-14      	51726506	       233.4 ns/op	     216 B/op	       2 allocs/op
BenchmarkCompareMedium/SjsonDecode-14         	 2090338	      5885 ns/op	    5924 B/op	     115 allocs/op
BenchmarkCompareMedium/StdDecode-14           	 1456482	      8071 ns/op	     504 B/op	      11 allocs/op
BenchmarkCompareMedium/JsoniterDecode-14      	 5981205	      2046 ns/op	     384 B/op	      41 allocs/op

第八轮：通过 []byte 优化 encode，提升性能
BenchmarkCompareMedium/SjsonMarshal-14         	35357608	       339.4 ns/op	     216 B/op	       2 allocs/op
BenchmarkCompareMedium/StdMarshal-14           	49267867	       244.6 ns/op	     216 B/op	       2 allocs/op
BenchmarkCompareMedium/JsoniterMarshal-14      	51075765	       238.1 ns/op	     216 B/op	       2 allocs/op
BenchmarkCompareMedium/SjsonUnmarshal-14       	 1948396	      5828 ns/op	    5924 B/op	     115 allocs/op
BenchmarkCompareMedium/StdUnmarshal-14         	 1469923	      8227 ns/op	     504 B/op	      11 allocs/op
BenchmarkCompareMedium/JsoniterUnmarshal-14    	 5809028	      2028 ns/op	     384 B/op	      41 allocs/op
```

## 4. 与其他 JSON 库的性能对比
测试命令：go test -bench=BenchmarkCompareMedium -benchmem -benchtime=10s -cpuprofile=cpu.prof -memprofile=mem.prof

```
第一轮：
BenchmarkCompareMedium/SjsonEncode-14         	 4837003	      2472 ns/op	    4073 B/op	      64 allocs/op
BenchmarkCompareMedium/StdEncode-14           	47469918	       251.2 ns/op	     216 B/op	       2 allocs/op
BenchmarkCompareMedium/JsoniterEncode-14      	50867079	       239.8 ns/op	     216 B/op	       2 allocs/op
BenchmarkCompareMedium/SjsonDecode-14         	 2031243	      5920 ns/op	    5924 B/op	     115 allocs/op
BenchmarkCompareMedium/StdDecode-14           	 1475060	      8225 ns/op	     504 B/op	      11 allocs/op
BenchmarkCompareMedium/JsoniterDecode-14      	 5912746	      2051 ns/op	     384 B/op	      41 allocs/op

第二轮：
BenchmarkCompareMedium/SjsonMarshal-14         	35773339	       329.0 ns/op	     216 B/op	       2 allocs/op
BenchmarkCompareMedium/StdMarshal-14           	50904760	       248.4 ns/op	     216 B/op	       2 allocs/op
BenchmarkCompareMedium/JsoniterMarshal-14      	48260007	       242.7 ns/op	     216 B/op	       2 allocs/op
BenchmarkCompareMedium/SjsonUnmarshal-14       	 2104412	      5856 ns/op	    5924 B/op	     115 allocs/op
BenchmarkCompareMedium/StdUnmarshal-14         	 1468572	      8210 ns/op	     504 B/op	      11 allocs/op
BenchmarkCompareMedium/JsoniterUnmarshal-14    	 5912791	      2068 ns/op	     352 B/op	      38 allocs/op

第三轮：
BenchmarkCompareMedium/SjsonMarshal-14         	37409500	       316.3 ns/op	     216 B/op	       2 allocs/op
BenchmarkCompareMedium/StdMarshal-14           	51325837	       241.2 ns/op	     216 B/op	       2 allocs/op
BenchmarkCompareMedium/JsoniterMarshal-14      	52111992	       229.6 ns/op	     216 B/op	       2 allocs/op
BenchmarkCompareMedium/SjsonUnmarshal-14       	 2724361	      4396 ns/op	     168 B/op	       6 allocs/op
BenchmarkCompareMedium/StdUnmarshal-14         	 1499334	      7980 ns/op	     504 B/op	      11 allocs/op
BenchmarkCompareMedium/JsoniterUnmarshal-14    	 6126705	      1995 ns/op	     352 B/op	      38 allocs/op
```

## 5. byte_utils.go 和 strconv 对比
测试命令：go test -bench=BenchmarkParseIntComparison -benchmem -benchtime=10s -cpuprofile=cpu.prof -memprofile=mem.prof
```
BenchmarkParseIntComparison/parseIntFromBytes-14         	984710859	        12.35 ns/op	       0 B/op	       0 allocs/op
BenchmarkParseIntComparison/strconv.ParseInt-14          	651232766	        19.15 ns/op	       0 B/op	       0 allocs/op
```

测试命令：go test -bench=BenchmarkParseFloatComparison -benchmem -benchtime=10s -cpuprofile=cpu.prof -memprofile=mem.prof
```
BenchmarkParseFloatComparison/parseFloatFromBytes-14         	886025876	        13.81 ns/op	       0 B/op	       0 allocs/op
BenchmarkParseFloatComparison/strconv.ParseFloat-14          	414038884	        28.85 ns/op	       0 B/op	       0 allocs/op
```