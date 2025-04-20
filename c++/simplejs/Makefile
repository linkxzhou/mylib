# SimpleJS Makefile
# 优化版本 - 更灵活的构建配置和更多功能

# 编译器和工具配置
CXX := g++
CXXFLAGS := -std=c++11 -Wall -Wextra -pthread
OPT_FLAGS := -O3
DEBUG_FLAGS := -g -DDEBUG

# 目标文件
LIB_NAME := simplejs
LIB_TARGET := lib$(LIB_NAME).a
SHARED_LIB := lib$(LIB_NAME).so
INCLUDE_DIRS := .

# 源文件和目标文件
SRCS := simplejs.cc
OBJS := $(SRCS:.cc=.o)
HEADERS := simplejs.h

# 测试配置
MAIN_TEST_SRC := main_test.cc
MAIN_TEST_BIN := main_test

# 安装配置
PREFIX := /usr/local
INSTALL_INCLUDE_DIR := $(PREFIX)/include
INSTALL_LIB_DIR := $(PREFIX)/lib

# 构建类型（默认为发布版本）
BUILD_TYPE ?= release

ifeq ($(BUILD_TYPE),debug)
  CXXFLAGS += $(DEBUG_FLAGS)
  BUILD_DIR := build/debug
else
  CXXFLAGS += $(OPT_FLAGS)
  BUILD_DIR := build/release
endif

# 创建构建目录下的目标文件路径
BUILD_OBJS := $(addprefix $(BUILD_DIR)/,$(OBJS))

# 增加 include 路径
CXXFLAGS += $(addprefix -I,$(INCLUDE_DIRS))

# 所有可用目标
.PHONY: all clean test main_test shared install uninstall help

# 默认目标：构建静态库
all: dirs $(BUILD_DIR)/$(LIB_TARGET)

# 创建必要的目录
dirs:
	@mkdir -p $(BUILD_DIR)

# 构建规则
$(BUILD_DIR)/%.o: %.cc $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/$(LIB_TARGET): $(BUILD_OBJS)
	$(AR) rcs $@ $^

# 构建共享库
shared: dirs $(BUILD_DIR)/$(SHARED_LIB)

$(BUILD_DIR)/$(SHARED_LIB): $(BUILD_OBJS)
	$(CXX) -shared -o $@ $^ $(LDFLAGS)

# 主测试程序
test: dirs $(BUILD_DIR)/$(MAIN_TEST_BIN)
	$(BUILD_DIR)/$(MAIN_TEST_BIN)

$(BUILD_DIR)/$(MAIN_TEST_BIN): $(MAIN_TEST_SRC) $(BUILD_DIR)/$(LIB_TARGET)
	$(CXX) $(CXXFLAGS) $< -L$(BUILD_DIR) -l$(LIB_NAME) -o $@

# 安装库和头文件
install: all
	@mkdir -p $(INSTALL_INCLUDE_DIR)
	@mkdir -p $(INSTALL_LIB_DIR)
	cp $(HEADERS) $(INSTALL_INCLUDE_DIR)/
	cp $(BUILD_DIR)/$(LIB_TARGET) $(INSTALL_LIB_DIR)/
	@if [ -f $(BUILD_DIR)/$(SHARED_LIB) ]; then \
		cp $(BUILD_DIR)/$(SHARED_LIB) $(INSTALL_LIB_DIR)/; \
	fi

# 卸载库和头文件
uninstall:
	rm -f $(INSTALL_INCLUDE_DIR)/$(HEADERS)
	rm -f $(INSTALL_LIB_DIR)/$(LIB_TARGET)
	rm -f $(INSTALL_LIB_DIR)/$(SHARED_LIB)

# 清理生成的文件
clean:
	rm -rf build
	rm -f $(TEST_BIN) $(MAIN_TEST_BIN)

# 显示帮助信息
help:
	@echo "可用的目标:"
	@echo "  all        - 构建静态库 (默认)"
	@echo "  shared     - 构建共享库"
	@echo "  test       - 构建并运行测试程序"
	@echo "  install    - 安装库和头文件到 $(PREFIX)"
	@echo "  uninstall  - 卸载库和头文件"
	@echo "  clean      - 删除生成的文件"
	@echo "  help       - 显示此帮助信息"
	@echo ""
	@echo "配置选项:"
	@echo "  BUILD_TYPE=debug|release - 设置构建类型 (默认: release)"
	@echo "  PREFIX=/path             - 设置安装路径 (默认: /usr/local)"
	@echo ""
	@echo "例如: make BUILD_TYPE=debug main_test"
