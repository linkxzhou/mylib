# google log makefile
CC  = gcc
CXX = g++
GCCVER := $(shell $(CC) -dumpversion | awk -F. '{ print $$1"."$$2}')
ifeq ($(shell uname -s), Darwin)
    CXX = clang++
	CFLAGS += -Wno-nullability-completeness -Wno-expansion-to-defined
endif

TARGET = glog.a
INC += -I./base -I./ -I../
LIB += -lpthread -ldl -lrt
OBJ = $(patsubst %.cpp, %.o, $(wildcard *.cpp))

OBJ_EXT = .o
CXXSRC_EXT = .cpp
CSRC_EXT = .c

ifeq ($(ARCH),32)
	CFLAGS += -m32 -march=pentium4
endif

CFLAGS  += -g -Wno-write-strings -Werror -MMD -D_GNU_SOURCE -D_REENTRANT -DSTACKSTRACE $(INC)
CXXFLAGS += $(CFLAGS)

$(TARGET): $(OBJ)
	@$(CXX) $(INC) $(LIB) $(CFLAGS) $(CPPFLAGS) -c $< -o $*.o
	@ar r $@ *.o
	@rm -f *.o *.d

.PHONY: all
all: $(TARGET)

.PHONY: clean
clean:
	@rm -f glog.a *.o