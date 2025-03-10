CXX = g++
CXXFLAGS = -Wall -fopenmp
TARGET = part3

all: $(TARGET)

$(TARGET): part3.o
	$(CXX) $(CXXFLAGS) -o $(TARGET) part3.o

part3.o: part3.cpp
	$(CXX) $(CXXFLAGS) -c part3.cpp -o part3.o

clean:
	rm -f $(TARGET) part3.o

run: $(TARGET)
	./$(TARGET) 5000 0.00001