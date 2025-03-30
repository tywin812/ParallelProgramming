#include <iostream>
#include <queue>
#include <future>
#include <thread>
#include <chrono>
#include <cmath>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <unordered_map>
#include <random>
#include <fstream>

template<typename T>
class Server {
    public:
        void start() {
            server_thread = std::jthread([this](std::stop_token stoken) {
                process_tasks(stoken);
            });
        }
    
        void stop() {
            server_thread.request_stop();
            queue_cv.notify_all();
        }
    
        size_t add_task(std::function<T()> task) {
            std::packaged_task<T()> packaged_task(std::move(task));
            auto future = packaged_task.get_future();
    
            size_t id = next_id++;
    
            {
                std::lock_guard lock(queue_mutex);
                tasks.emplace(id, std::move(packaged_task));
            }
    
            {
                std::lock_guard lock(results_mutex);
                results[id] = std::move(future);
            }
    
            queue_cv.notify_one();
            return id;
        }
    
        T request_result(size_t id) {
            std::unique_lock lock(results_mutex);
            auto it = results.find(id);
            if (it == results.end()) throw std::runtime_error("Task not found");
            lock.unlock();
    
            return it->second.get();
        }
    
    private:
        void process_tasks(std::stop_token stoken) {
            while (true) {
                std::unique_lock lock(queue_mutex);
                queue_cv.wait(lock, [&] { return !tasks.empty() || stoken.stop_requested(); });
    
                if (stoken.stop_requested()) break;
    
                auto task_pair = std::move(tasks.front());
                tasks.pop();
                lock.unlock();
    
                task_pair.second();
    
            }
        }
    
        std::jthread server_thread;
        size_t next_id = 0;
    
        std::queue<std::pair<size_t, std::packaged_task<T()>>> tasks;
        std::unordered_map<size_t, std::future<T>> results;
    
        std::mutex queue_mutex;
        std::condition_variable queue_cv;
        std::mutex results_mutex;
};
    

template <typename T>
T Tsin(T x) {
    return std::sin(x);
}

template <typename T>
T Tsqrt(T x) {
    return std::sqrt(x);
}

template <typename T>
T Tpow(T x, int y) {
    return std::pow(x, y);
}

template<typename T>
void client_sin(Server<T>& server, int N, const std::string& filename) {
    std::ofstream file(filename);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dist(-3.14, 3.14);

    for (int i = 0; i < N; ++i) {
        T arg = dist(gen);
        size_t id = server.add_task([arg] { return Tsin(arg); });
        T result = server.request_result(id);
        file << "sin(" << arg << ") = " << result << std::endl;
    }
}

template<typename T>
void client_sqrt(Server<T>& server, int N, const std::string& filename) {
    std::ofstream file(filename);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dist(0.0, 100.0);

    for (int i = 0; i < N; ++i) {
        T arg = dist(gen);
        size_t id = server.add_task([arg] { return Tsqrt(arg); });
        T result = server.request_result(id);
        file << "sqrt(" << arg << ") = " << result << std::endl;
    }
}

template<typename T>
void client_pow(Server<T>& server, int N, const std::string& filename) {
    std::ofstream file(filename);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dist_x(1.0, 10.0);
    std::uniform_int_distribution<int> dist_y(0, 5);

    for (int i = 0; i < N; ++i) {
        T x = dist_x(gen);
        int y = dist_y(gen);
        size_t id = server.add_task([x,y] { return Tpow(x,y); });
        T result = server.request_result(id);
        file << x << "^" << y << " = " << result << std::endl;
    }
}

int main() {
    Server<double> server;
    server.start();

    std::thread sin_client(client_sin<double>, std::ref(server), 1000, "sin_results.txt");
    std::thread sqrt_client(client_sqrt<double>, std::ref(server), 1000, "sqrt_results.txt");
    std::thread pow_client(client_pow<double>, std::ref(server), 1000, "pow_results.txt");

    sin_client.join();
    sqrt_client.join();
    pow_client.join();

    server.stop();
    return 0;
}