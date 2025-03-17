#include <omp.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <string>

const double epsilon = 1e-5;

void simpleIteration(const std::vector<std::vector<double>>& A, const std::vector<double>& b, std::vector<double>& x, double t) {
    int N = A.size();
    std::vector<double> Ax(N);
    std::vector<double> diff(N);
    std::fill(x.begin(), x.end(), 0.0);

    double start_time = omp_get_wtime();

    double L2_b = 0.0;
    for (int i = 0; i < N; ++i) {
        L2_b += b[i] * b[i];
    }
    
    double L2_norm = 0.0;

    while(true) {
        for (int i = 0; i < N; ++i) {
            Ax[i] = 0.0;
            for (int j = 0; j < N; ++j) {
                Ax[i] += A[i][j] * x[j];
            }
        }

        for (int i = 0; i < N; ++i) {
            diff[i] = Ax[i] - b[i];
        }

        L2_norm = 0.0;
        for (int i = 0; i < N; ++i) {
            L2_norm += diff[i] * diff[i];
        }

        if (std::sqrt(L2_norm) / std::sqrt(L2_b) < epsilon) {
            break;
        }

        for (int i = 0; i < N; ++i) {
            x[i] -= t * diff[i];
        }
    }
    double end_time = omp_get_wtime(); 
    std::cout << "Simple Iteration (serial) Time: " << end_time - start_time << " seconds\n";
}

void simpleItFors(const std::vector<std::vector<double>>& A, const std::vector<double>& b, std::vector<double>& x, double t, int threads) {
    int N = A.size();
    std::vector<double> Ax(N);
    std::vector<double> diff(N);
    std::fill(x.begin(), x.end(), 0.0);
    double L2_b = 0.0;
    double L2_norm = 0.0;

    double start_time = omp_get_wtime();

    #pragma omp parallel for reduction(+:L2_b)
    for (int i = 0; i < N; ++i) {
        L2_b += b[i] * b[i];
    }

    while(true) {
        #pragma omp parallel for num_threads(threads)
        for (int i = 0; i < N; ++i) {
            Ax[i] = 0.0;
            for (int j = 0; j < N; ++j) {
                Ax[i] += A[i][j] * x[j];
            }
        }

        #pragma omp parallel for num_threads(threads)
        for (int i = 0; i < N; ++i) {
            diff[i] = Ax[i] - b[i];
        }

        #pragma omp single
        L2_norm = 0.0;
        
        #pragma omp parallel for reduction(+:L2_norm)
        for (int i = 0; i < N; ++i) {
            L2_norm += diff[i] * diff[i];
        }

        if (std::sqrt(L2_norm) / std::sqrt(L2_b) < epsilon) {
            break;
        }

        #pragma omp parallel for num_threads(threads)
        for (int i = 0; i < N; ++i) {
            x[i] -= t * diff[i];
        }
    }
    double end_time = omp_get_wtime(); 
    std::cout << "1st solution. " <<  "Threads:" << threads << ". Simple Iteration (parallel fors) Time: " << end_time - start_time << " seconds\n";
}

void simpleItParallel(const std::vector<std::vector<double>>& A, const std::vector<double>& b, std::vector<double>& x, double t, int threads) {
    int N = A.size();
    std::vector<double> Ax(N);
    std::vector<double> diff(N);
    std::fill(x.begin(), x.end(), 0.0);
    double L2_b = 0.0;
    double L2_norm = 0.0;
    bool stop = false;  

    double start_time = omp_get_wtime();

    #pragma omp parallel num_threads(threads)
    {
        #pragma omp for reduction(+:L2_b)
        for (int i = 0; i < N; ++i) {
            L2_b += b[i] * b[i];
        }

        while (true) {
            #pragma omp for
            for (int i = 0; i < N; ++i) {
                Ax[i] = 0.0;
                for (int j = 0; j < N; ++j) {
                    Ax[i] += A[i][j] * x[j];
                }
            }

            #pragma omp for
            for (int i = 0; i < N; ++i) {
                diff[i] = Ax[i] - b[i];
            }

            #pragma omp single
            L2_norm = 0.0;

            #pragma omp for reduction(+:L2_norm)
            for (int i = 0; i < N; ++i) {
                L2_norm += diff[i] * diff[i];
            }

            #pragma omp single
            {
                if (std::sqrt(L2_norm) / std::sqrt(L2_b) < epsilon) {
                    stop = true;
                }
            }

            if (stop) break;

            #pragma omp for
            for (int i = 0; i < N; ++i) {
                x[i] -= t * diff[i];
            }
        }
    }

    double end_time = omp_get_wtime();
    std::cout << "1st solution. " << "Threads:" << threads << ". Simple Iteration (parallel section) Time: " << end_time - start_time << " seconds\n";
}



int main(int argc, char** argv) {
    int N = std::stoi(argv[1]);
    double t = std::stod(argv[2]);

    std::vector<std::vector<double>> A(N, std::vector<double>(N, 1.0));
    for (int i = 0; i < N; ++i) {
        A[i][i] = 2.0; 
    }

    std::vector<double> b(N, N + 1.0); 
    std::vector<double> x(N, 0.0); 

    simpleIteration(A, b, x, t);

    std::cout << std::endl;

    simpleItFors(A, b, x, t, 2);
    simpleItFors(A, b, x, t, 4);
    simpleItFors(A, b, x, t, 7);
    simpleItFors(A, b, x, t, 8);
    simpleItFors(A, b, x, t, 16);
    simpleItFors(A, b, x, t, 20);
    simpleItFors(A, b, x, t, 40);

    std::cout << std::endl;

    simpleItParallel(A, b, x, t, 2);
    simpleItParallel(A, b, x, t, 4);
    simpleItParallel(A, b, x, t, 7);
    simpleItParallel(A, b, x, t, 8);
    simpleItParallel(A, b, x, t, 16);
    simpleItParallel(A, b, x, t, 20);
    simpleItParallel(A, b, x, t, 40);

    return 0;
}