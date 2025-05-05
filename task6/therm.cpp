#include <iostream>
#include <cmath>
#include <chrono>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

void initialize_boundaries(double* x, int N) {
    x[0] = 10.0;
    x[N - 1] = 20.0;
    x[N * N - 1] = 30.0;
    x[N * (N - 1)] = 20.0;

    for (int j = 0; j < N; ++j) {
        x[j] = 10.0 + (20.0 - 10.0) * j / (N - 1);                  // Верх
        x[(N - 1) * N + j] = 30.0 + (20.0 - 30.0) * j / (N - 1);    // Низ
    }

    for (int i = 0; i < N; ++i) {
        x[i * N] = 10.0 + (20.0 - 10.0) * i / (N - 1);              // Лево
        x[i * N + (N - 1)] = 20.0 + (30.0 - 20.0) * i / (N - 1);    // Право
    }
}

void solve(int N, double epsilon, int max_iter) {
    double* x = new double[N * N]();
    double* x_new = new double[N * N]();
    initialize_boundaries(x, N);

    int iter = 0;
    double residual = 0.0;
    auto start = std::chrono::high_resolution_clock::now();

    #pragma acc data copy(x[0:N*N]) create(x_new[0:N*N])
    {
        do {
            residual = 0.0;

            #pragma acc parallel loop collapse(2) reduction(+:residual)
            for (int i = 1; i < N - 1; ++i) {
                for (int j = 1; j < N - 1; ++j) {
                    int idx = i * N + j;
                    double updated = 0.25 * (x[(i - 1) * N + j] + x[(i + 1) * N + j] +
                                            x[i * N + (j - 1)] + x[i * N + (j + 1)]);
                    residual += (updated - x[idx]) * (updated - x[idx]);
                    x_new[idx] = updated;
                }
            }

            #pragma acc parallel loop collapse(2)
            for (int i = 1; i < N - 1; ++i) {
                for (int j = 1; j < N - 1; ++j) {
                    int idx = i * N + j;
                    x[idx] = x_new[idx];
                }
            }

            residual = std::sqrt(residual);
            iter++;
        } while (residual > epsilon && iter < max_iter);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Iterations: " << iter << ", Residual: " << residual
              << ", Time: " << elapsed.count() << " s\n";

    if (N == 10 || N == 13) {
        std::cout << "Final grid:\n";
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                std::cout << x[i * N + j] << "\t";
            }
            std::cout << "\n";
        }
    }

    delete[] x;
    delete[] x_new;
}

int main(int argc, char* argv[]) {
    int N;
    double epsilon;
    int max_iter;

    try {
        po::options_description desc("Allowed options");
        desc.add_options()
            ("help,h", "produce help message")
            ("size,N", po::value<int>(&N)->default_value(128), "grid size (N)")
            ("epsilon,e", po::value<double>(&epsilon)->default_value(1e-6), "epsilon")
            ("max_iter,m", po::value<int>(&max_iter)->default_value(1000000), "maximum iterations");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);

        if (vm.count("help")) {
            std::cout << desc << "\n";
            return 1;
        }

        solve(N, epsilon, max_iter);

    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}