# CMakeLists.txt

cmake_minimum_required(VERSION 2.8.12)
project(bindings)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories( ${EIGEN3_INCLUDE_DIRS} )

pybind11_add_module(bindings bindings.cpp)