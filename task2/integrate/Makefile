CXX = g++
CXXFLAGS = -Wall -fopenmp
TARGET = part2

all: $(TARGET)

$(TARGET): part2.o
	$(CXX) $(CXXFLAGS) -o $(TARGET) part2.o

part2.o: part2.cpp
	$(CXX) $(CXXFLAGS) -c part2.cpp -o part2.o

clean:
	rm -f $(TARGET) part2.o

run: $(TARGET)
	./$(TARGET)