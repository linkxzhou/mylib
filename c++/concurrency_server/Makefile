# Makefile for Concurrency Server Models

CXX = g++
CXXFLAGS = -std=c++11 -Wall -Wextra -O2 -pthread
TARGET = concurrency_server
SOURCES = main.cpp base/event_dispatcher.cpp base/coroutine.cpp
HEADERS = base/server_base.h \
          base/event_dispatcher.h \
          base/select_dispatcher.h \
          base/poll_dispatcher.h \
          base/epoll_dispatcher.h \
          base/kqueue_dispatcher.h \
          single_process_server.h \
          multi_process_server.h \
          multi_thread_server.h \
          process_pool1_server.h \
          process_pool2_server.h \
          thread_pool_server.h \
          leader_follower_server.h \
          select_server.h \
          poll_server.h \
          epoll_server.h \
          kqueue_server.h \
          reactor_server.h \
          coroutine_server.h

# 检测操作系统
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    CXXFLAGS += -DLINUX
endif
ifeq ($(UNAME_S),Darwin)
    CXXFLAGS += -DMACOS
endif

.PHONY: all clean test help

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES)

clean:
	rm -f $(TARGET)

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