#include <omp.h>
#include <cmath>
#include <iostream>

double func(double x)
{
    return std::exp(-x * x);
}

double integrate(double (*func)(double), double a, double b, int n)
{
    double h = (b - a) / n;
    double sum = 0.0;

    for (int i = 0; i < n; i++)
    sum += func(a + h * (i + 0.5));
    sum *= h;
    return sum;
}

double integrate_omp(double (*func)(double), double a, double b, int n, int threads)
{
    double h = (b - a) / n;
    double sum = 0.0;

    #pragma omp parallel num_threads(threads)
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = n / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (n - 1) : (lb + items_per_thread - 1);
        double sumloc = 0.0;
        
        for (int i = lb; i <= ub; i++)
            sumloc += func(a + h * (i + 0.5));

        #pragma omp atomic
        sum += sumloc;
    }
    
    sum *= h;
    return sum;
}

void run_serial(double (*func)(double), double a, double b, int nsteps)
{
    double start_time = omp_get_wtime();
    double res = integrate(func, a, b, nsteps);
    double end_rime = omp_get_wtime();
    std::cout << "Elapsed time (serial): " << end_rime - start_time << " sec. " << "Result: " << res << " Error:" << std::abs(res - std::sqrt(M_PI)) << std::endl;
}

void run_parallel(double (*func)(double), double a, double b, int nsteps, int threads)
{
    double start_time = omp_get_wtime();
    double res = integrate_omp(func, a, b, nsteps, threads);
    double end_rime = omp_get_wtime();
    std::cout << "Threads: " << threads << ". Elapsed time (parallel): " << end_rime - start_time << " sec. " << "Result: " << res << ". Error:" << std::abs(res - std::sqrt(M_PI)) << std::endl;
}

int main()
{
    const double a = -4.0;
    const double b = 4.0;
    const int nsteps = 40000000;
    std::cout << "Integration of exp(-x^2) from " << a << " to " << b << ", nsteps = " << nsteps << std::endl;

    run_serial(func, a, b, nsteps);
    run_parallel(func, a, b, nsteps, 2);
    run_parallel(func, a, b, nsteps, 4);
    run_parallel(func, a, b, nsteps, 7);
    run_parallel(func, a, b, nsteps, 8);
    run_parallel(func, a, b, nsteps, 16);
    run_parallel(func, a, b, nsteps, 20);
    run_parallel(func, a, b, nsteps, 40);
    
    return 0;
}