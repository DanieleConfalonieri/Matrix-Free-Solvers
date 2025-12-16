/**
 * @file main.cc
 * @brief Main entry point for the Matrix-Free ADR Solver application.
 */

#include "ADROperator.hpp"
#include "MatrixFreeSolver.hpp"

/**
 * @brief Main function.
 *
 * Initializes the MPI environment and runs the Matrix-Free solver for a specific test case.
 *
 * **Test Case Configuration:**
 * - **Domain:** Hypercube (Cube in 3D).
 * - **Problem:** Pure Diffusion (Poisson-like) with constant forcing.
 * - \f$ \mu = 1.0 \f$ (Diffusion coefficient)
 * - \f$ \beta = [0, 0, 0] \f$ (Advection field disabled)
 * - \f$ \gamma = 0.0 \f$ (Reaction coefficient)
 * - \f$ f = 1.0 \f$ (Constant source term)
 * - **Boundary Conditions:**
 * - Homogeneous Dirichlet (\f$ u=0 \f$) on all 6 faces of the cube (IDs 0-5).
 * - No Neumann boundaries active.
 *
 * @param argc Number of command line arguments.
 * @param argv Command line arguments array.
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

        // Define Advection vector (currently zero for pure diffusion test)
        std::vector<double> beta_vector(dimension, 0.0); 
        // beta_vector[0] = 1.0; // Uncomment to set beta_x = 1.0
        
        // Instantiate the solver with constant coefficients
        MatrixFreeSolver<dimension,
                        degree_finite_element,
                        double> matrix_free_solver(
                            std::make_shared<Functions::ConstantFunction<dimension, double>>(1.0), // mu
                            std::make_shared<Functions::ConstantFunction<dimension, double>>(beta_vector), // beta
                            std::make_shared<Functions::ConstantFunction<dimension, double>>(0.0), // gamma
                            std::make_shared<Functions::ConstantFunction<dimension, double>>(1.0), // forcing
                            std::make_shared<Functions::ConstantFunction<dimension, double>>(0.0),  // neumann value (unused)
                            std::make_shared<Functions::ConstantFunction<dimension, double>>(0.0),  // dirichlet value
                            std::set<types::boundary_id>{0,1,2,3,4,5}, // Apply Dirichlet to all boundaries
                            std::set<types::boundary_id>{}  // No Neumann boundaries
                        );

        matrix_free_solver.run();
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