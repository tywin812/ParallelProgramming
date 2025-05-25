#include <iostream> 
#include <cmath>
#include <chrono>
#include <vector>
#include <cub/block/block_reduce.cuh>
#include <cub/cub.cuh>
#include <boost/program_options.hpp>
#include <nvtx3/nvToolsExt.h>

namespace po = boost::program_options;

#define BLOCK_THREADS 512
#define CHECK_INTERVAL 1000

void initialize_boundaries(double* x, int N) {
    x[0] = 10.0;
    x[N - 1] = 20.0;
    x[N * N - 1] = 30.0;
    x[N * (N - 1)] = 20.0;

    for (int j = 0; j < N; ++j) {
        x[j] = 10.0 + (20.0 - 10.0) * j / (N - 1);
        x[(N - 1) * N + j] = 30.0 + (20.0 - 30.0) * j / (N - 1);
    }

    for (int i = 0; i < N; ++i) {
        x[i * N] = 10.0 + (20.0 - 10.0) * i / (N - 1);
        x[i * N + (N - 1)] = 20.0 + (30.0 - 20.0) * i / (N - 1);
    }
}

__global__ void update_kernel(double* d_x, double* d_x_new, double* d_diff, int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (i < N-1 && j < N-1) {
        int idx = i * N + j;
        d_x_new[idx] = 0.25 * (d_x[(i - 1) * N + j] + d_x[(i + 1) * N + j] +
                                d_x[i * N + (j - 1)] + d_x[i * N + (j + 1)]);
        d_diff[idx] = fabs(d_x_new[idx] - d_x[idx]);
    }
}

__global__ void copy_kernel(double* d_x, double* d_x_new, int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (i < N-1 && j < N-1) {
        int idx = i * N + j;
        d_x[idx] = d_x_new[idx];
    }
}

__global__ void block_max_kernel(double* d_input, int total, double* d_output, int num_blocks) {
    typedef cub::BlockReduce<double, BLOCK_THREADS> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    BlockReduce reduce_inst(temp_storage);
    int idx = threadIdx.x + blockIdx.x * BLOCK_THREADS;
    double val = (idx < total) ? d_input[idx] : -__builtin_inf();
    double block_max = reduce_inst.Reduce(val, cub::Max());
    if (threadIdx.x == 0) {
        d_output[blockIdx.x] = block_max;
    }
}


void solve(int N, double epsilon, int max_iter) {
    double* x = new double[N * N]();
    initialize_boundaries(x, N);

    
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    double *d_x, *d_x_new, *d_diff;
    cudaMallocAsync((void**)&d_x, N * N * sizeof(double), stream); 
    cudaMallocAsync((void**)&d_x_new, N * N * sizeof(double), stream);
    cudaMallocAsync((void**)&d_diff, (N-2) * (N-2) * sizeof(double),stream);
    cudaMemcpy(d_x, x, N * N * sizeof(double), cudaMemcpyHostToDevice);

    int total = (N-2)*(N-2);
    std::vector<int> temp_sizes;
    int current_size = total;
    while (current_size > 1) {
        int num_blocks = (current_size + BLOCK_THREADS - 1) / BLOCK_THREADS;
        temp_sizes.push_back(num_blocks);
        current_size = num_blocks;
    }

    double* d_temp_buffer;
    size_t temp_buffer_size = 0;
    for (auto s : temp_sizes) temp_buffer_size += s;
    cudaMallocAsync(&d_temp_buffer, temp_buffer_size * sizeof(double), stream);

    std::vector<double*> temp_arrays(temp_sizes.size());
    size_t offset = 0;
    for (size_t i = 0; i < temp_sizes.size(); ++i) {
        temp_arrays[i] = d_temp_buffer + offset;
        offset += temp_sizes[i];
    }

    dim3 block_update(16, 16);
    dim3 grid_update((N-2 + block_update.x - 1) / block_update.x, (N-2 + block_update.y - 1) / block_update.y);

    cudaGraph_t graph;
    cudaGraphCreate(&graph, 0);

    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    nvtxRangePushA("update_kernel");
    update_kernel<<<grid_update, block_update, 0, stream>>>(d_x, d_x_new, d_diff, N);
    nvtxRangePop();

    double* current_array = d_diff;
    current_size = total;

    nvtxRangePushA("block_max_kernel_reduction");
    for (size_t i = 0; i < temp_sizes.size(); ++i) {
        int num_blocks = temp_sizes[i];
        block_max_kernel<<<num_blocks, BLOCK_THREADS, 0, stream>>>(current_array, current_size, temp_arrays[i], num_blocks);
        current_array = temp_arrays[i];
        current_size = num_blocks;
    }
    nvtxRangePop();

    nvtxRangePushA("copy_kernel");
    copy_kernel<<<grid_update, block_update, 0, stream>>>(d_x, d_x_new, N);
    cudaStreamEndCapture(stream, &graph);
    nvtxRangePop();

    cudaGraphExec_t instance;
    cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);

    int iter = 0;
    double residual = std::numeric_limits<double>::max();
    auto start = std::chrono::high_resolution_clock::now();

    nvtxRangePushA("solving");
    double max_error = 0.0;

    while (residual > epsilon && iter < max_iter) {
        for (int k = 0; k < CHECK_INTERVAL && iter < max_iter; ++k) {
            cudaGraphLaunch(instance, stream);
            iter++;
        }

        cudaMemcpyAsync(&max_error, temp_arrays.back(), sizeof(double), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);  
        residual = max_error;
    }
    nvtxRangePop();

    auto end = std::chrono::high_resolution_clock::now();

    cudaMemcpy(x, d_x, N * N * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_x_new);
    cudaFree(d_diff);
    cudaFree(d_temp_buffer);
    cudaGraphExecDestroy(instance);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Итераций: " << iter << ", Остаточная ошибка: " << residual
              << ", Время: " << elapsed.count() << " с\n";

    if (N == 10 || N == 13) {
        std::cout << "Итоговая сетка:\n";
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                std::cout << x[i * N + j] << "\t";
            }
            std::cout << "\n";
        }
    }

    delete[] x;
}

int main(int argc, char* argv[]) {
    nvtxRangePushA("Init context");
    cudaSetDevice(3);
    nvtxRangePop();

    int N;
    double epsilon;
    int max_iter;

    try {
        po::options_description desc("Допустимые опции");
        desc.add_options()
            ("help,h", "показать справку")
            ("size,N", po::value<int>(&N)->default_value(128), "размер сетки (N)")
            ("epsilon,e", po::value<double>(&epsilon)->default_value(1e-6), "эпсилон")
            ("max_iter,m", po::value<int>(&max_iter)->default_value(1000000), "максимум итераций");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);

        if (vm.count("help")) {
            std::cout << desc << "\n";
            return 1;
        }
        solve(N, epsilon, max_iter);

    } catch (std::exception& e) {
        std::cerr << "Ошибка: " << e.what() << "\n";
        return 1;
    }

    return 0;
}