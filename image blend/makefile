# Makefile for OpenCV C++ program

# Compiler and compiler flags
CXX = g++
CXXFLAGS = -std=c++11

# OpenCV configuration using pkg-config
OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4)
OPENCV_LIBS := $(shell pkg-config --libs opencv4)

# Source files and output executable
SOURCES = Q4.cpp
EXECUTABLE = Q4

# Compiler flags for additional libraries
CXXFLAGS += -fopenmp 


# Linker flags
LDFLAGS = -lm -fopenmp

$(EXECUTABLE): $(SOURCES)
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $(EXECUTABLE) $(OPENCV_CFLAGS) $(OPENCV_LIBS) $(LDFLAGS)

clean:
	rm -f $(EXECUTABLE)

.PHONY: all clean
