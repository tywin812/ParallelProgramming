CXX = g++
CXXFLAGS = -Wall -O2
TARGET = task1

all: $(TARGET)

$(TARGET): task1.o
	$(CXX) $(CXXFLAGS) -o $(TARGET) task1.o

task1.o: task1.cpp
	$(CXX) $(CXXFLAGS) -c task1.cpp -o task1.o

clean:
	rm -f $(TARGET) task1.o

run-float: $(TARGET)
	./$(TARGET) float

run-double: $(TARGET)
	./$(TARGET) double
