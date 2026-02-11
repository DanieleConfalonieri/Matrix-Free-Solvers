/**
 * @file main.cpp
 * @brief Minimal driver for a matrix-free ADR test.
 */

#include "ADROperator.hpp"
#include "MatrixFreeSolver.hpp"
#include "MatrixFreeSolverMG.hpp"

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

        const unsigned int dimension = 3;
        const unsigned int degree_finite_element = 2;
                const bool enableMultigrid = true; // Set to false to disable multigrid preconditioning

        const unsigned int mesh_refinement_level = (argc > 1) ? std::stoi(argv[1]) : 4; // Adjust from command line for finer/coarser mesh

        // Advection vector (example: non-zero x component)
        std::vector<double> beta_vector(dimension, 0.0);
        beta_vector[0] = 1.0; // Example: set beta_x = 1.0
        
        // Instantiate the solver with constant coefficients
        if(enableMultigrid){
          MatrixFreeSolverMG<dimension,
                        degree_finite_element,
                        double> matrix_free_solver_mg(
                            std::make_shared<Functions::ConstantFunction<dimension, double>>(1.0), // Mu
                            std::make_shared<Functions::ConstantFunction<dimension, double>>(beta_vector), // Beta
                            std::make_shared<Functions::ConstantFunction<dimension, double>>(0.0), // Gamma
                            std::make_shared<Functions::ConstantFunction<dimension, double>>(1.0), // Forcing
                            std::make_shared<Functions::ConstantFunction<dimension, double>>(0.0),  // Neumann value
                            std::make_shared<Functions::ConstantFunction<dimension, double>>(0.0),  // Dirichlet value
                            std::set<types::boundary_id>{0,1,2,3,4,5}, // Apply Dirichlet to all boundaries
                            std::set<types::boundary_id>{}  // Apply Neumann to all boundaries
                        );
          matrix_free_solver_mg.run();
        } else {
          MatrixFreeSolver<dimension,
                        degree_finite_element,
                        double> matrix_free_solver(
                            std::make_shared<Functions::ConstantFunction<dimension, double>>(1.0), // Mu
                            std::make_shared<Functions::ConstantFunction<dimension, double>>(beta_vector), // Beta
                            std::make_shared<Functions::ConstantFunction<dimension, double>>(0.0), // Gamma
                            std::make_shared<Functions::ConstantFunction<dimension, double>>(1.0), // Forcing
                            std::make_shared<Functions::ConstantFunction<dimension, double>>(0.0),  // Neumann value
                            std::make_shared<Functions::ConstantFunction<dimension, double>>(0.0),  // Dirichlet value
                            std::set<types::boundary_id>{0,1,2,3,4,5}, // Apply Dirichlet to all boundaries
                            std::set<types::boundary_id>{}  // Apply Neumann to all boundaries
                        );
          matrix_free_solver.run();
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