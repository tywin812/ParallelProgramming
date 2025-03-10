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

void simpleItSchedule(const std::vector<std::vector<double>>& A, const std::vector<double>& b, std::vector<double>& x, double t, int threads, const std::string& schedule_type,int chunk_size) {
    int N = A.size();
    std::vector<double> Ax(N);
    std::vector<double> diff(N);
    std::fill(x.begin(), x.end(), 0.0);
    double L2_b = 0.0;
    double L2_norm = 0.0;

    if (schedule_type == "static") 
    {
        omp_set_schedule(omp_sched_static, chunk_size);
    }
    else if (schedule_type == "dynamic") {
        omp_set_schedule(omp_sched_dynamic, chunk_size);
    } 
    else if (schedule_type == "guided") {
        omp_set_schedule(omp_sched_guided, chunk_size);
    }

    double start_time = omp_get_wtime();

    #pragma omp parallel for reduction(+:L2_b) schedule(runtime)
    for (int i = 0; i < N; ++i) {
        L2_b += b[i] * b[i];
    }

    while(true) {
        #pragma omp parallel for num_threads(threads) schedule(runtime)
        for (int i = 0; i < N; ++i) {
            Ax[i] = 0.0; 
            for (int j = 0; j < N; ++j) {
                Ax[i] += A[i][j] * x[j];
            }
        }

        #pragma omp parallel for num_threads(threads) schedule(runtime)
        for (int i = 0; i < N; ++i) {
            diff[i] = Ax[i] - b[i];
        }

        L2_norm = 0.0;
        #pragma omp parallel for reduction(+:L2_norm) schedule(runtime)
        for (int i = 0; i < N; ++i) {
            L2_norm += diff[i] * diff[i];
        }

        if (std::sqrt(L2_norm) / std::sqrt(L2_b) < epsilon) {
            break;
        }

        #pragma omp parallel for num_threads(threads) schedule(runtime)
        for (int i = 0; i < N; ++i) {
            x[i] -= t * diff[i];
        }
    }
    double end_time = omp_get_wtime(); 
    std::cout << "Threads: " << threads << ", Schedule: " << schedule_type << "(" << chunk_size << ")"
              << ", Time: " << (end_time - start_time) << " sec\n";
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

    simpleItSchedule(A, b, x, t, 4, "static", 1250);
    simpleItSchedule(A, b, x, t, 4, "dynamic", 1250);
    simpleItSchedule(A, b, x, t, 4, "guided", 1250);

    return 0;
}