CXX = g++
CXXFLAGS = -Wall -std=c++20 -pthread
TARGET = tests

all: $(TARGET)

$(TARGET): tests.o
	$(CXX) $(CXXFLAGS) -o $(TARGET) tests.o

part2.o: tests.cpp
	$(CXX) $(CXXFLAGS) -c tests.cpp -o tests.o

clean:
	rm -f $(TARGET) tests.o

run-tests: $(TARGET)
	./$(TARGET)