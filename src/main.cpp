/**
 * @file main.cpp
 * @brief Minimal driver for a matrix-free ADR test.
 */

#include "ADROperator.hpp"
#include "MatrixFreeSolver.hpp"
#include "MatrixFreeSolverMG.hpp"
#include <cstring>

/**
 * @brief Run a single predefined test: 3D cube, diffusion, constant source.
 *
 * Minimal entry point: initializes MPI and runs one solver instance.
 * @return 0 on success, 1 on exception.
 */
int main (int argc, char *argv[])
{
        using namespace dealii;

        // Initialize MPI. The '1' argument limits console output to valid ASCII if needed.
        Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

        ConditionalOStream pcout(std::cout,
          (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0));

        if(argc == 2 && std::strcmp(argv[1], "--help") == 0){
          pcout << "First parameter: profiling_run (1 --> just for profiling, anything else --> generates output)" << std::endl;
          pcout << "Second parameter: mesh_refinement_level (default 4)" << std::endl;
          pcout << "Third parameter: enable_multigrid (-mg --> enable multigrid preconditioner, anything else --> disable multigrid preconditioner)" << std::endl << std::endl;
          pcout << "Example usage: mpirun -n 4 " << argv[0] << " 1 6 -mg" << std::endl;
          pcout << "This runs the solver with 4 process, without output generation, a global refinement level of 6, and the multigrid preconditioner enabled." << std::endl << std::endl;
          pcout << "If you want to run the executable using the default parameters you MUST run it without using ANY command line parameter" << std::endl;
          pcout << "Example usage with default parameters: mpirun -n 4 " << argv[0] << std::endl;
          pcout << "This runs the solver with 4 process, with output generation enabled, a global refinement level of 4, and the multigrid preconditioner disabled." << std::endl << std::endl;
          return 0;
        }

        const bool profiling_run = (argc > 1) ? (std::stoi(argv[1]) == 1) : false; // Set to true to disable output file (.vtk) generation for profiling runs

        const unsigned int mesh_refinement_level = (argc > 2) ? std::stoi(argv[2]) : 4; // Adjust from command line for finer/coarser mesh

        const bool enable_multigrid = (argc > 3) ? (std::strcmp(argv[3], "-mg") == 0) : false; // Set to true to enable multigrid preconditioning if "-mg" flag is provided

        const unsigned int dimension = 3;
        const unsigned int degree_finite_element = 2;

        // Problem coefficients and BCs (shared by MG and non-MG solvers)
        const std::vector<double> beta_components = {1.0, -1.0, 1.0};
        const auto mu     = std::make_shared<Functions::ConstantFunction<dimension, double>>(3.0);
        const auto beta   = std::make_shared<Functions::ConstantFunction<dimension, double>>(beta_components);
        const auto gamma  = std::make_shared<Functions::ConstantFunction<dimension, double>>(2.0);
        const auto forcing = std::make_shared<Functions::ConstantFunction<dimension, double>>(1.0);
        const auto h      = std::make_shared<Functions::ConstantFunction<dimension, double>>(2.0);
        const auto g      = std::make_shared<Functions::ConstantFunction<dimension, double>>(7.0);
        const std::set<types::boundary_id> dirichlet_ids = {1, 2, 3, 4};
        const std::set<types::boundary_id> neumann_ids   = {0, 5};

        if (enable_multigrid) {
          MatrixFreeSolverMG<dimension, degree_finite_element, double> solver(
              mu, beta, gamma, forcing, h, g, dirichlet_ids, neumann_ids, mesh_refinement_level);
          solver.run(profiling_run);
        } else {
          MatrixFreeSolver<dimension, degree_finite_element, double> solver(
              mu, beta, gamma, forcing, h, g, dirichlet_ids, neumann_ids, mesh_refinement_level);
          solver.run(profiling_run);
        }
  return 0;
}