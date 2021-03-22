
#include <pybind11/pybind11.h>
#include <eigen3/Eigen/Dense>

#include "solvers/puct.hpp"
#include "problems/example1.hpp"

struct search(
    Eigen::Matrix<float,2,1> state;
)
{
    Example1 m_problem;
    PUCT m_puct;
    auto result = m_puct.search(state); 
    return result; 
}

PYBIND11_MODULE(bindings, m) {
    m.def("cpp_search", &search, "PUCT");
}