CXX = g++
CXXFLAGS = -Wall -fopenmp
TARGET = part1

all: $(TARGET)

$(TARGET): part1.o
	$(CXX) $(CXXFLAGS) -o $(TARGET) part1.o

part1.o: part1.cpp
	$(CXX) $(CXXFLAGS) -c part1.cpp -o part1.o

clean:
	rm -f $(TARGET) part1.o

run-20k: $(TARGET)
	./$(TARGET) 20000 20000

run-40k: $(TARGET)
	./$(TARGET) 40000 40000