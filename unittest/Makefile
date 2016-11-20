# google log makefile
# author by linkxzhou@tencent.com
CC  = gcc
CXX = g++
GCCVER := $(shell $(CC) -dumpversion | awk -F. '{ print $$1"."$$2}')

TARGET = unittest
INC += -I../ -I../base
LIB += ../glog.a -lpthread -ldl -lrt
OBJ = $(patsubst %.cpp, %.o, $(wildcard logging_striptest_main.cpp))

OBJ_EXT = .o
CXXSRC_EXT = .cpp
CSRC_EXT = .c

ifeq ($(ARCH),32)
	CFLAGS += -m32 -march=pentium4
endif

CFLAGS  += -g -D_SPP_PROXY -D_MP_MODE -Wno-write-strings -Werror -MMD -D_GNU_SOURCE -D_REENTRANT -DSTACKSTRACE
CXXFLAGS+=$(CFLAGS)

$(TARGET): $(OBJ)
	@$(CXX) -c $(INC) $(LIB) $(CFLAGS) $(CPPFLAGS)

.PHONY: all
all: $(TARGET)

.PHONY: clean
clean:
	@rm -f *.o # - means: ignore error