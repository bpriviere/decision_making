# CMakeLists.txt

cmake_minimum_required(VERSION 2.8.12)
project(bindings)

find_package(Eigen3 REQUIRED NO_MODULE)

include_directories(
  ${EIGEN3_INCLUDE_DIRS}
)

find_package(pybind11 REQUIRED)
pybind11_add_module(bindings bindings.cpp)