cmake_minimum_required(VERSION 3.16)

project(matrix_free_adr)
include(${CMAKE_SOURCE_DIR}/common/cmake-common.cmake)

set(CMAKE_CXX_STANDARD 20) # Required for <concepts>
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Complete MatrixFree solver
add_executable(adr_solver
    src/main.cpp                 # Il main della simulazione
    src/MatrixFreeSolver.cpp     # L'implementazione del solver (Decommentato!)
    src/MatrixFreeSolverMG.cpp   # Implementation of MG-enabled solver
    src/ADROperator.cpp          # L'implementazione dell'operatore
    
    # Headers (Opzionali per la compilazione, utili per farli vedere agli IDE)
    src/MatrixFreeSolver.hpp
    src/ADROperator.hpp
)

deal_ii_setup_target(adr_solver)

# Complete MatrixBased Solver
add_executable(matrix_based
                src/exercise-01-parallel.cpp
                src/DiffusionReaction-parallel.cpp
# Headers
src/DiffusionReaction-parallel.hpp)
deal_ii_setup_target(matrix_based)

#Testing operator
#add_executable(operator_test
#    src/example1_matrix_free.cpp # Il main di test che avevamo fatto prima
#    src/ADROperator.cpp          # L'operatore serve anche qui!
#    
#    src/ADROperator.hpp
#)

#deal_ii_setup_target(operator_test)
