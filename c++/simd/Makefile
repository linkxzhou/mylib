CXX = g++

# 检测架构并设置相应的编译标志
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_M),arm64)
    # ARM64 (Apple Silicon) 编译标志
    CXXFLAGS = -std=c++17 -O3 -mcpu=native -Wall -Wextra
else
    # x86_64 编译标志
    CXXFLAGS = -std=c++17 -O3 -march=native -mavx2 -msse4.2 -Wall -Wextra
endif

TARGETS = test benchmark main

all: $(TARGETS)

test: test.cpp simd_strlen.h
	$(CXX) $(CXXFLAGS) -o test test.cpp

benchmark: benchmark.cpp simd_strlen.h
	$(CXX) $(CXXFLAGS) -o benchmark benchmark.cpp

main: main.cpp simd_strlen.h
	$(CXX) $(CXXFLAGS) -o main main.cpp

run_test: test
	./test

run_benchmark: benchmark
	./benchmark

clean:
	rm -f $(TARGETS)

.PHONY: all clean run_test run_benchmark