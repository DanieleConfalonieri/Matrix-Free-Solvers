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
  try
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

        // Advection vector (example: non-zero x component)
        std::vector<double> beta_vector(dimension, 0.0);
        beta_vector[0] = 1.0; // Example: set beta_x = 1.0
        beta_vector[1] = -1.0; // beta_y = 0.0
        beta_vector[2] = 1.0; // beta_z = 0.0
        
        // Instantiate the solver with constant coefficients
        if(enable_multigrid){
          printf("Running with MG preconditioner enabled.\n");
          MatrixFreeSolverMG<dimension,
                        degree_finite_element,
                        double> matrix_free_solver_mg(
                            std::make_shared<Functions::ConstantFunction<dimension, double>>(3.0), // Mu
                            std::make_shared<Functions::ConstantFunction<dimension, double>>(beta_vector), // Beta
                            std::make_shared<Functions::ConstantFunction<dimension, double>>(2.0), // Gamma
                            std::make_shared<Functions::ConstantFunction<dimension, double>>(1.0), // Forcing
                            std::make_shared<Functions::ConstantFunction<dimension, double>>(2.0),  // Neumann value
                            std::make_shared<Functions::ConstantFunction<dimension, double>>(7.0),  // Dirichlet value
                            std::set<types::boundary_id>{1,2,3,4}, // Apply Dirichlet to all boundaries
                            std::set<types::boundary_id>{0,5},  // Apply Neumann to all boundaries
                            mesh_refinement_level // Global refinement level for the initial mesh
                        );
          matrix_free_solver_mg.run(profiling_run);
        } else {
          MatrixFreeSolver<dimension,
                        degree_finite_element,
                        double> matrix_free_solver(
                            std::make_shared<Functions::ConstantFunction<dimension, double>>(3.0), // Mu
                            std::make_shared<Functions::ConstantFunction<dimension, double>>(beta_vector), // Beta
                            std::make_shared<Functions::ConstantFunction<dimension, double>>(2.0), // Gamma
                            std::make_shared<Functions::ConstantFunction<dimension, double>>(1.0), // Forcing
                            std::make_shared<Functions::ConstantFunction<dimension, double>>(2.0),  // Neumann value
                            std::make_shared<Functions::ConstantFunction<dimension, double>>(7.0),  // Dirichlet value
                            std::set<types::boundary_id>{1,2,3,4}, // Apply Dirichlet to all boundaries
                            std::set<types::boundary_id>{0,5},  // Apply Neumann to all boundaries
                            mesh_refinement_level // Global refinement level for the initial mesh
                        );
          matrix_free_solver.run(profiling_run);
        }
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl;
      std::cerr << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  return 0;
}