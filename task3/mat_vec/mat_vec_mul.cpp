#include <vector>
#include <thread>
#include <string>
#include <chrono>
#include <iostream>
#include <functional>


void initialize(std::vector<double>& matrix,std::vector<double>& vec, int start, int end, int size){
    for (int i = start; i < end; i++){
        for (int j = 0; j < size; j++){
            matrix[i*size + j] = i+j;
        }
        vec[i] = i;
    }
}

void matrix_vector_mul(std::vector<double>& matrix,std::vector<double>& vec, std::vector<double>& result, int start, int end, int size){
    for (int i = start; i < end; i++){
        result[i] = 0;
        for (int j = 0; j < size; j++){
            result[i] += matrix[i*size+j]*vec[j];
        }
    }
}

void parallel_init(std::vector<double>& matrix,std::vector<double>& vec,std::vector<std::jthread>& threads, int num_threads, int size){
    threads.reserve(num_threads);
    int chunk_size = size/num_threads;
    int remaining = size % num_threads;

    for (int i = 0; i < num_threads; i++){
        int start = i * chunk_size;
        int end = start + chunk_size;
        if (i == num_threads - 1){
            end += remaining;
        }
        threads.emplace_back(initialize, std::ref(matrix), std::ref(vec), start, end, size);
    }
}

void parallel_mul(std::vector<double>& matrix,std::vector<double>& vec,std::vector<double>& result, std::vector<std::jthread>& threads, int num_threads, int size){
    threads.reserve(num_threads);
    int chunk_size = size/num_threads;
    int remaining = size % num_threads;

    for (int i = 0; i < num_threads; i++){
        int start = i * chunk_size;
        int end = start + chunk_size;
        if (i == num_threads - 1){
            end += remaining;
        }
        threads.emplace_back(matrix_vector_mul, std::ref(matrix), std::ref(vec), std::ref(result), start, end, size);
    }
}

void run_serial(int size){
    std::vector<double> matrix(size*size);
    std::vector<double> vec(size);
    std::vector<double> result(size, 0.0);

    auto start_time = std::chrono::high_resolution_clock::now();
    initialize(matrix, vec, 0, size, size);
    matrix_vector_mul(matrix, vec, result, 0, size, size);
    auto end_time = std::chrono::high_resolution_clock::now();

    const std::chrono::duration<double> time{end_time - start_time};

    std::cout << "Serial: " << time.count() << std::endl;
}

void run_parallel(int size, int num_threads){
    std::vector<double> matrix(size*size);
    std::vector<double> vec(size);
    std::vector<double> result(size, 0.0);
    std::vector<std::jthread> threads;

    auto start_time = std::chrono::high_resolution_clock::now();
    {
        std::vector<std::jthread> init_threads;
        parallel_init(matrix, vec, init_threads, num_threads, size);
    }

    {
        std::vector<std::jthread> mul_threads;
        parallel_mul(matrix, vec, result, mul_threads, num_threads, size);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();

    const std::chrono::duration<double> time{end_time - start_time};
    std::cout << "Parallel(" << num_threads << " threads): " << time.count() << std::endl; 
}

int main(int argc, char** argv){
    int size = std::stoi(argv[1]);

    run_serial(size);
    run_parallel(size, 2);
    run_parallel(size, 4);
    run_parallel(size, 7);
    run_parallel(size, 8);
    run_parallel(size, 16);
    run_parallel(size, 20);
    run_parallel(size, 40);

    return 0;
}