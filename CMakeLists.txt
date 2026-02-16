cmake_minimum_required(VERSION 3.16)

project(matrix_free_adr)
include(${CMAKE_SOURCE_DIR}/common/cmake-common.cmake)

set(CMAKE_CXX_STANDARD 20) # Required for <concepts>
set(CMAKE_CXX_STANDARD_REQUIRED ON)

### VALIDATION

# Complete MatrixFree solver
add_executable(
    validation_free
    src/validation_free.cpp
    src/MatrixFreeSolver.cpp
    src/MatrixFreeSolverMG.cpp
    src/ADROperator.cpp
    
    # Headers (Opzionali per la compilazione, utili per farli vedere agli IDE)
    src/MatrixFreeSolver.hpp
    src/ADROperator.hpp
)

deal_ii_setup_target(validation_free)

# Complete MatrixBased Solver
add_executable(
    validation_based
    src/validation_based.cpp
    src/DiffusionReaction-parallel.cpp

    # Headers
    src/DiffusionReaction-parallel.hpp
)

deal_ii_setup_target(validation_based)

### PROFILING

# Complete MatrixFree solver
add_executable(
    profiling_free
    src/profiling_free.cpp
    src/MatrixFreeSolver.cpp
    src/MatrixFreeSolverMG.cpp
    src/ADROperator.cpp
    
    # Headers (Opzionali per la compilazione, utili per farli vedere agli IDE)
    src/MatrixFreeSolver.hpp
    src/ADROperator.hpp
)

deal_ii_setup_target(profiling_free)

# Complete MatrixBased Solver
add_executable(
    profiling_based
    src/profiling_based.cpp
    src/DiffusionReaction-parallel.cpp

    # Headers
    src/DiffusionReaction-parallel.hpp
)

deal_ii_setup_target(profiling_based)


#Testing operator
#add_executable(operator_test
#    src/example1_matrix_free.cpp # Il main di test che avevamo fatto prima
#    src/ADROperator.cpp          # L'operatore serve anche qui!
#    
#    src/ADROperator.hpp
#)

#deal_ii_setup_target(operator_test)
