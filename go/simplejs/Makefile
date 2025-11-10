# Makefile for SimpleJS v2.0 (C++11 version) - Simplified

CXX = g++
CXXFLAGS = -std=c++11 -Wall -Wextra -O2 -g
INCLUDES = -I.

# Source files
MAIN_TEST_SRC = main_test.cpp
HEADER_V2 = simplejs_v2.h

# Target executables
MAIN_TEST_BIN = main_test
MAIN_TEST_DEBUG_BIN = main_test_debug

# Default target
all: test

# Build test program
$(MAIN_TEST_BIN): $(MAIN_TEST_SRC) $(HEADER_V2)
	@echo "Building SimpleJS v2.0 test..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $(MAIN_TEST_SRC)
	@echo "✅ Build completed: $@"

# Build debug version
$(MAIN_TEST_DEBUG_BIN): $(MAIN_TEST_SRC) $(HEADER_V2)
	@echo "Building SimpleJS v2.0 test (debug)..."
	$(CXX) $(CXXFLAGS) -DDEBUG -O0 $(INCLUDES) -o $@ $(MAIN_TEST_SRC)
	@echo "✅ Debug build completed: $@"

# Run tests
test: $(MAIN_TEST_BIN)
	@echo "\n🚀 Running SimpleJS v2.0 tests..."
	@echo "==========================================="
	./$(MAIN_TEST_BIN)
	@echo "==========================================="
	@echo "✅ Tests completed!"

# Run debug tests
debug: $(MAIN_TEST_DEBUG_BIN)
	@echo "\n🚀 Running SimpleJS v2.0 tests (debug)..."
	@echo "============================================="
	./$(MAIN_TEST_DEBUG_BIN)
	@echo "============================================="
	@echo "✅ Debug tests completed!"

# Clean build artifacts
clean:
	@echo "🧹 Cleaning build artifacts..."
	rm -f $(MAIN_TEST_BIN) $(MAIN_TEST_DEBUG_BIN)
	@echo "✅ Clean completed!"

# Show help
help:
	@echo "SimpleJS v2.0 Build System (Simplified)"
	@echo "========================================"
	@echo "Available targets:"
	@echo "  all    - Build test program (default)"
	@echo "  test   - Build and run tests"
	@echo "  debug  - Build and run debug tests"
	@echo "  clean  - Clean build artifacts"
	@echo "  help   - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make test   # Build and run tests"
	@echo "  make debug  # Build and run debug tests"

# Phony targets
.PHONY: all test debug clean help

# Default goal
.DEFAULT_GOAL := all