# Makefile for Concurrency Server Models

CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -pthread -Ibase
LDFLAGS = -lssl -lcrypto

# 目标文件
TARGET = server
UNIFIED_TARGET = unified_server
EXAMPLE_TARGET = example_server

# 源文件
MAIN_SOURCES = main.cpp
BASE_SOURCES = base/server_factory.cpp base/enhanced_event_loop.cpp base/connection_manager.cpp \
               base/protocol_handler.cpp base/monitoring.cpp base/event_dispatcher.cpp
EXAMPLE_SOURCES = example_unified_server.cpp

# 对象文件
BASE_OBJECTS = $(BASE_SOURCES:.cpp=.o)
MAIN_OBJECTS = $(MAIN_SOURCES:.cpp=.o)
EXAMPLE_OBJECTS = $(EXAMPLE_SOURCES:.cpp=.o)

# 默认目标
all: $(TARGET) $(UNIFIED_TARGET) $(EXAMPLE_TARGET)

# 原始服务器（兼容性）
$(TARGET): $(MAIN_OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(MAIN_OBJECTS) $(LDFLAGS)

# 统一服务器库
$(UNIFIED_TARGET): $(BASE_OBJECTS)
	ar rcs lib$(UNIFIED_TARGET).a $(BASE_OBJECTS)

# 示例服务器
$(EXAMPLE_TARGET): $(EXAMPLE_OBJECTS) $(BASE_OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(EXAMPLE_TARGET) $(EXAMPLE_OBJECTS) $(BASE_OBJECTS) $(LDFLAGS)

# 编译规则
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# 安装基础库
install: $(UNIFIED_TARGET)
	mkdir -p /usr/local/include/concurrency_server
	cp -r base/*.h /usr/local/include/concurrency_server/
	cp lib$(UNIFIED_TARGET).a /usr/local/lib/
	ldconfig

# 测试目标
test: $(EXAMPLE_TARGET)
	./$(EXAMPLE_TARGET) &
	echo "Server started, testing..."
	sleep 2
	curl -s http://localhost:8080/ > /dev/null && echo "HTTP test passed" || echo "HTTP test failed"
	curl -s http://localhost:8080/api/stats > /dev/null && echo "Stats API test passed" || echo "Stats API test failed"
	killall $(EXAMPLE_TARGET) 2>/dev/null || true

# 性能测试（需要安装wrk）
benchmark: $(EXAMPLE_TARGET)
	./$(EXAMPLE_TARGET) &
	echo "Starting benchmark..."
	sleep 3
	command -v wrk >/dev/null 2>&1 && wrk -t4 -c100 -d10s http://localhost:8080/ || echo "wrk not found, skipping benchmark"
	killall $(EXAMPLE_TARGET) 2>/dev/null || true

# 内存检查（需要安装valgrind）
memcheck: $(EXAMPLE_TARGET)
	valgrind --leak-check=full --show-leak-kinds=all ./$(EXAMPLE_TARGET)

# 代码格式化
format:
	find . -name "*.cpp" -o -name "*.h" | xargs clang-format -i

# 静态分析
analyze:
	cppcheck --enable=all --std=c++17 .

# 文档生成
docs:
	doxygen Doxyfile 2>/dev/null || echo "Doxygen not found, skipping documentation generation"

# 清理
clean:
	rm -f $(TARGET) $(EXAMPLE_TARGET) lib$(UNIFIED_TARGET).a
	rm -f $(BASE_OBJECTS) $(MAIN_OBJECTS) $(EXAMPLE_OBJECTS)
	rm -rf docs/html docs/latex

# 深度清理
distclean: clean
	rm -f *.log core.*

# 帮助信息
help:
	@echo "Available targets:"
	@echo "  all        - Build all targets"
	@echo "  server     - Build original server (compatibility)"
	@echo "  unified_server - Build unified server library"
	@echo "  example_server - Build example server"
	@echo "  install    - Install library to system"
	@echo "  test       - Run basic functionality tests"
	@echo "  benchmark  - Run performance benchmark"
	@echo "  memcheck   - Run memory leak check"
	@echo "  format     - Format code with clang-format"
	@echo "  analyze    - Run static code analysis"
	@echo "  docs       - Generate documentation"
	@echo "  clean      - Remove build artifacts"
	@echo "  distclean  - Remove all generated files"
	@echo "  help       - Show this help message"

.PHONY: all install test benchmark memcheck format analyze docs clean distclean help

# 测试不同的服务器模型
test-single: $(TARGET)
	./$(TARGET) single_process 8080

test-multi-process: $(TARGET)
	./$(TARGET) multi_process 8080

test-multi-thread: $(TARGET)
	./$(TARGET) multi_thread 8080

test-process-pool1: $(TARGET)
	./$(TARGET) process_pool1 8080

test-process-pool2: $(TARGET)
	./$(TARGET) process_pool2 8080

test-thread-pool: $(TARGET)
	./$(TARGET) thread_pool 8080

test-leader-follower: $(TARGET)
	./$(TARGET) leader_follower 8080

test-select: $(TARGET)
	./$(TARGET) select 8080

test-poll: $(TARGET)
	./$(TARGET) poll 8080

test-epoll: $(TARGET)
	./$(TARGET) epoll 8080

test-kqueue: $(TARGET)
	./$(TARGET) kqueue 8080

test-reactor: $(TARGET)
	./$(TARGET) reactor 8080

test-coroutine: $(TARGET)
	./$(TARGET) coroutine 8080

test-work-stealing: $(TARGET)
	./$(TARGET) work_stealing 8080

test-actor: $(TARGET)
	./$(TARGET) actor 8080

test-fiber: $(TARGET)
	./$(TARGET) fiber 8080

test-producer-consumer: $(TARGET)
	./$(TARGET) producer_consumer 8080

test-half-sync-async: $(TARGET)
	./$(TARGET) half_sync_async 8080

test-proactor: $(TARGET)
	./$(TARGET) proactor 8080

test-pipeline: $(TARGET)
	./$(TARGET) pipeline 8080

test-hybrid: $(TARGET)
	./$(TARGET) hybrid 8080

# 测试所有模型
test-all: test-single test-multi-thread test-multi-process test-process-pool1 test-process-pool2 test-thread-pool test-leader-follower test-select test-poll test-epoll test-kqueue test-reactor test-coroutine test-work-stealing test-actor test-fiber test-producer-consumer test-half-sync-async test-proactor test-pipeline test-hybrid

# 性能测试（需要安装ab工具）
bench: $(TARGET)
	@echo "Starting server in background..."
	./$(TARGET) thread_pool 8080 &
	@sleep 2
	@echo "Running benchmark..."
	ab -n 1000 -c 10 http://localhost:8080/
	@echo "Stopping server..."
	pkill -f $(TARGET)

help:
	@echo "Available targets:"
	@echo "  all              - Build the server"
	@echo "  clean            - Remove built files"
	@echo "  test-<model>     - Test specific server model"
	@echo "  bench            - Run performance benchmark"
	@echo "  help             - Show this help"
	@echo ""
	@echo "Available server models:"
	@echo "  single_process, multi_process, multi_thread"
	@echo "  process_pool1, process_pool2, thread_pool"
	@echo "  leader_follower, select, poll, epoll, kqueue"
	@echo "  reactor, coroutine"
	@echo ""
	@echo "Usage: ./$(TARGET) <model> [port]"
	@echo "Example: ./$(TARGET) thread_pool 8080"