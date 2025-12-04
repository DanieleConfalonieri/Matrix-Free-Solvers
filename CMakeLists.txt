cmake_minimum_required(VERSION 3.12.0)
project(project_title LANGUAGES CXX C)

# Da CMakeLists.txt a common/ è: common/..., NON ../common/...
include(common/cmake-common.cmake)

# Eseguibile che compila solo MatrixFreeSolver.cpp
add_executable(MatrixFreeSolver
    src/MatrixFreeSolver.cpp
)

# Macro definita in cmake-common.cmake (deal.II, flags, include, ecc.)
deal_ii_setup_target(MatrixFreeSolver)
