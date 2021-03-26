#include <pybind11/pybind11.h>

// c++ -O3 -Wall -shared -std=c++11 -fPIC -I/home/ben/anaconda3/envs/dm_env/include/python3.7m -I/home/ben/anaconda3/envs/dm_env/lib/python3.7/site-packages/pybind11/include testing/example.cpp -o example.cpython-37m-x86_64-linux-gnu.so

int add(int i, int j) {
    return i + j;
}

PYBIND11_MODULE(example, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("add", &add, "A function which adds two numbers");
}
