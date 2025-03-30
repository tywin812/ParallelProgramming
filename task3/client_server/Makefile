CXX = g++
CXXFLAGS = -Wall -std=c++20 -pthread
TARGET = client_server

all: $(TARGET)

$(TARGET): client_server.o
	$(CXX) $(CXXFLAGS) -o $(TARGET) client_server.o

part2.o: client_server.cpp
	$(CXX) $(CXXFLAGS) -c client_server.cpp -o client_server.o

clean:
	rm -f $(TARGET) client_server.o

run: $(TARGET)
	./$(TARGET)