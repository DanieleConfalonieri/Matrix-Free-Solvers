cmake_minimum_required(VERSION 3.16)

project(matrix_free_adr)
include(${CMAKE_SOURCE_DIR}/common/cmake-common.cmake)


set(CMAKE_CXX_STANDARD 20) #Required for <concepts>
set(CMAKE_CXX_STANDARD_REQUIRED ON)


add_executable(adr_solver
    src/example1_matrix_free.cpp
    src/ADROperator.cpp
    #src/MatrixFreeSolver.cpp
    #header files (for IDEs)
    src/ADROperator.hpp 
)

deal_ii_setup_target(adr_solver)
