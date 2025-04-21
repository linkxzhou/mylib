SHELL = /bin/sh
CXX = g++
CXXFLAGS = -std=c++11 -Wall -Wextra -O2 -Wno-reorder -Wno-unused-parameter -Wno-format -DDEBUG
LDFLAGS = -lpthread
INCLUDE = -I.

SRC = httpd.cc
OBJ = $(SRC:.cc=.o)
TARGET = tinyhttpd
HEADERS = httpd.h

all: $(TARGET)

$(TARGET): $(OBJ) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -o $@ $(OBJ) $(LDFLAGS)

%.o: %.cc $(HEADERS)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -o $@

clean:
	rm -f *.o $(TARGET)

# Debug build
debug: CXXFLAGS += -DDEBUG -g

debug: clean all

.PHONY: all clean debug