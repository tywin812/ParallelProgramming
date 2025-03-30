#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <string>
#include <stdexcept>

bool compare_double(double a, double b, double epsilon = 1e-5) {
    return std::fabs(a - b) < epsilon;
}

void test_results(const std::string& filename, const std::string& operation) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        double x, y, result;
        char op;

        if (operation == "sin") {
            if (iss >> op >> x >> op >> op >> result) {
                double expected = std::sin(x);
                if (!compare_double(result, expected)) {
                    std::cerr << "Error in " << filename << ": sin(" << x << ") = " << result << ", expected " << expected << std::endl;
                }
            }
        } else if (operation == "sqrt") {
            if (iss >> op >> x >> op >> op >> result) {
                double expected = std::sqrt(x);
                if (!compare_double(result, expected)) {
                    std::cerr << "Error in " << filename << ": sqrt(" << x << ") = " << result << ", expected " << expected << std::endl;
                }
            }
        } else if (operation == "pow") {
            if (iss >> x >> op >> y >> op >> result) {
                double expected = std::pow(x, static_cast<int>(y));
                if (!compare_double(result, expected, 0.3)) {
                    std::cerr << "Error in " << filename << ": " << x << "^" << y << " = " << result << ", expected " << expected << std::endl;
                }
            }
        }
    }
}

int main() {
    try {
        test_results("../sin_results.txt", "sin");
        test_results("../sqrt_results.txt", "sqrt");
        test_results("../pow_results.txt", "pow");
        std::cout << "Test completed! Check error messages above." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}