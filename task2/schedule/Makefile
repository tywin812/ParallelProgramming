CXX = g++
CXXFLAGS = -Wall -fopenmp
TARGET = schedule

all: $(TARGET)

$(TARGET): schedule.o
	$(CXX) $(CXXFLAGS) -o $(TARGET) schedule.o

part3.o: schedule.cpp
	$(CXX) $(CXXFLAGS) -c schedule.cpp -o schedule.o

clean:
	rm -f $(TARGET) schedule.o

run: $(TARGET)
	./$(TARGET) 5000 0.00001
