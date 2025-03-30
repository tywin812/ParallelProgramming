CXX = g++
CXXFLAGS = -Wall -std=c++20 -pthread
TARGET = mat_vec_mul

all: $(TARGET)

$(TARGET): mat_vec_mul.o
	$(CXX) $(CXXFLAGS) -o $(TARGET) mat_vec_mul.o

part2.o: mat_vec_mul.cpp
	$(CXX) $(CXXFLAGS) -c mat_vec_mul.cpp -o mat_vec_mul.o

clean:
	rm -f $(TARGET) mat_vec_mul.o

run-20k: $(TARGET)
	./$(TARGET) 20000

run-40k: $(TARGET)
	./$(TARGET) 40000