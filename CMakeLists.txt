cmake_minimum_required(VERSION 3.16)
project(matrix_free_adr)
include(${CMAKE_SOURCE_DIR}/common/cmake-common.cmake)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Libreria Matrix-Free
add_library(adr_core_free STATIC
    src/MatrixFreeSolver.cpp
    src/MatrixFreeSolverMG.cpp
    src/ADROperator.cpp
)
deal_ii_setup_target(adr_core_free)

# Libreria Matrix-Based
add_library(adr_core_based STATIC
    src/DiffusionReaction-parallel.cpp
)
deal_ii_setup_target(adr_core_based)


### VALIDATION (Homogeneous)
add_executable(validation_free src/validation_free.cpp)
target_link_libraries(validation_free adr_core_free)

add_executable(validation_based src/validation_based.cpp)
target_link_libraries(validation_based adr_core_based)

### VALIDATION (Inhomogeneous)
add_executable(validation_inhomogeneous_free src/validation_inhomogeneous_free.cpp)
target_link_libraries(validation_inhomogeneous_free adr_core_free)

add_executable(validation_inhomogeneous_based src/validation_inhomogeneous_based.cpp)
target_link_libraries(validation_inhomogeneous_based adr_core_based)

### PROFILING
add_executable(profiling_free src/profiling_free.cpp)
target_link_libraries(profiling_free adr_core_free)

add_executable(profiling_based src/profiling_based.cpp)
target_link_libraries(profiling_based adr_core_based)

### WEAK SCALING
add_executable(weak_scaling_free src/weak_scaling_free.cpp)
target_link_libraries(weak_scaling_free adr_core_free)

add_executable(weak_scaling_based src/weak_scaling_based.cpp)
target_link_libraries(weak_scaling_based adr_core_based)

### calc_core
add_executable(calc_core build/calc_core.cpp)