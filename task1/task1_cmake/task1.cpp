#include <iostream>
#include <cmath>
#include <string>
#define SIZE 10000000

template <typename T>
T SumSin() {
    T* arr = new T[SIZE];
    const double period = M_PI;       
    const double rad = period / (SIZE / 2);
    double cur_rad = 0.0;      

    for (int i = 0; i < SIZE; i++) {
        arr[i] = static_cast<T>(std::sin(cur_rad)); 
        cur_rad += rad;
    }

    T sum = 0;
    for (int i = 0; i < SIZE; i++) {
        sum += arr[i];
    }

    delete[] arr;
    return sum;
}

int main(int argc, char** argv)
{
    std::string type(argv[1]);

    if (type == "float")
    {
        float sum = SumSin<float>();
        std::cout << "Sum (float): " << sum << std::endl;
    }
    else if (type == "double")
    {
        double sum = SumSin<double>();
        std::cout << "Sum (double): " << sum << std::endl;
    }
    else
    {
        std::cerr << "Invalid type. Use 'float' or 'double'." << std::endl;
        return 1;
    }

    return 0;
}