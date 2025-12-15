#include "ADROperator.hpp"
#include "MatrixFreeSolver.hpp"

int main (int argc, char *argv[])
{
  try
    {
        using namespace dealii;
        Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

        const unsigned int dimension = 3;
        const unsigned int degree_finite_element = 2;

        std::vector<double> beta_vector(dimension, 0.0); 
        //beta_vector[0] = 1.0; // Example: set beta_x = 1.0, beta_y = 0.0, beta_z = 0.0
        
        MatrixFreeSolver<dimension,
                        degree_finite_element,
                        double> matrix_free_solver(
                            std::make_shared<Functions::ConstantFunction<dimension, double>>(1.0), // mu
                            std::make_shared<Functions::ConstantFunction<dimension, double>>(beta_vector), // beta
                            std::make_shared<Functions::ConstantFunction<dimension, double>>(0.0), // gamma
                            std::make_shared<Functions::ConstantFunction<dimension, double>>(1.0), // forcing
                            std::make_shared<Functions::ConstantFunction<dimension, double>>(0.0),  // neumann
                            std::make_shared<Functions::ConstantFunction<dimension, double>>(0.0),  // dirichlet
                            std::set<types::boundary_id>{0,1,2,3,4,5}, // dirichlet boundary ids
                            std::set<types::boundary_id>{}  // neumann boundary ids
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