#include <vector>
#include <omp.h>
#include <iostream>
#include <string> 

void matrix_vector_product(const std::vector<double>& matrix, const std::vector<double>& vector, std::vector<double>& result, int m, int n)
{
    for (int i = 0; i < m; i++) {
        double sum = 0.0;
        for (int j = 0; j < n; j++) {
            sum += matrix[i * n + j] * vector[j];
        }
        result[i] = sum;
    }
}

void matrix_vector_product_omp(const std::vector<double>& matrix, const std::vector<double>& vector, std::vector<double>& result, int threads, int m, int n)
{
    #pragma omp parallel for num_threads(threads)
    for (int i = 0; i < m; i++) {
        double sum = 0.0;
        for (int j = 0; j < n; j++) {
            sum += matrix[i * n + j] * vector[j];
        }
        result[i] = sum;
    }
}

void run_serial(int m, int n){
    std::vector<double> matrix(m*n);
    std::vector<double> vector(n);
    std::vector<double> result(m, 0.0);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i * n + j] = i + j;
        }
    }

    for (int j = 0; j < n; j++){
        vector[j] = j;
    }

    double start_time = omp_get_wtime();
    matrix_vector_product(matrix, vector, result, m, n);
    double end_time = omp_get_wtime();
    std::cout << "Elapsed time (serial): " << end_time - start_time << " sec." << std::endl;
}

void run_parallel(int m, int n, int threads)
{
    std::vector<double> matrix(m*n);
    std::vector<double> vector(n);
    std::vector<double> result(m, 0.0);

    #pragma omp parallel for num_threads(threads)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i * n + j] = i + j;
        }
    }

    for (int j = 0; j < n; j++){
        vector[j] = j;
    }

    double start_time = omp_get_wtime();
    matrix_vector_product_omp(matrix, vector, result, threads, m, n);
    double end_time = omp_get_wtime();
    std::cout << "Matrix size: " << m << ". Elapsed time (parallel) for " << threads << " threads: "  << end_time - start_time << " sec." << std::endl;
}

int main(int argc, char** argv)
{
    int m = std::stoi(argv[1]);
    int n = std::stoi(argv[2]);
    
    // run_serial(m, n);
    // run_parallel(m, n, 2);
    // run_parallel(m, n, 4);
    // run_parallel(m, n, 7);
    // run_parallel(m, n, 8);
    // run_parallel(m, n, 16);
    run_parallel(m, n, 20);
    run_parallel(m, n, 40);

    return 0;
}