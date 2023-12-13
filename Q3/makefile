# Makefile for compiling Q3.cpp

# Compiler
CXX = g++

# Source file
SOURCE = Q3.cpp

# Executable name
TARGET = Q3

# Compiler flags
CXXFLAGS = -std=c++11 -O3 -march=native -fopenmp


# Linker flags
LDFLAGS = -lm -fopenmp

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) $(SOURCE) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
