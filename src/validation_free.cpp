/**
 * @file main.cpp
 * @brief Minimal driver for a matrix-free ADR test.
 */

#include "ADROperator.hpp"
#include "MatrixFreeSolver.hpp"
#include "MatrixFreeSolverMG.hpp"
#include <cstring>
#include <cmath>

using namespace dealii;

template <int dim>
class ForcingFunction : public Function<dim>
{
public:
  virtual double value(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
  {
    const double x = p[0];
    const double y = p[1];
    
    // FORMULA CORRETTA (metodo MMS)
    const double term_diff_react = (20.0 * M_PI * M_PI + 2.0) * std::cos(2.0 * M_PI * x) * std::cos(4.0 * M_PI * y);
    const double term_adv_x = -2.0 * M_PI * std::sin(2.0 * M_PI * x) * std::cos(4.0 * M_PI * y);
    const double term_adv_y = -4.0 * M_PI * std::cos(2.0 * M_PI * x) * std::sin(4.0 * M_PI * y);
    
    return term_diff_react + term_adv_x + term_adv_y;
  }
};

/**
 * @brief Run a single predefined test: 3D cube, diffusion, constant source.
 *
 * Minimal entry point: initializes MPI and runs one solver instance.
 * @return 0 on success, 1 on exception.
 */
int main (int argc, char *argv[])
{
        // Initialize MPI. The '1' argument limits console output to valid ASCII if needed.
        Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

        ConditionalOStream pcout(std::cout,
          (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0));

        if(argc == 2 && std::strcmp(argv[1], "--help") == 0){
          pcout << "First parameter: profiling_run (1 --> just for profiling, anything else --> generates output)" << std::endl;
          pcout << "Second parameter: mesh_refinement_level (default 4)" << std::endl;
          pcout << "Third parameter: enable_multigrid (-mg --> enable multigrid preconditioner, anything else --> disable multigrid preconditioner)" << std::endl << std::endl;
          pcout << "Fourth parameter: fe_degree (default 2) --> polynomial degree of the FE_Q finite element space used for the discretization" << std::endl << std::endl;
          pcout << "Example usage: mpirun -n 4 " << argv[0] << " 1 5 -mg 3" << std::endl;
          pcout << "This runs the solver with 4 process, without output generation, a global refinement level of 5, the multigrid preconditioner enabled and polynomial degree of 3." << std::endl << std::endl;
          pcout << "If you want to run the executable using the default parameters you MUST run it without using ANY command line parameter" << std::endl;
          pcout << "Example usage with default parameters: mpirun -n 4 " << argv[0] << std::endl;
          pcout << "This runs the solver with 4 process, with output generation enabled, a global refinement level of 4, the multigrid preconditioner disabled and polynomial degree of 2." << std::endl << std::endl;
          return 0;
        }

        const bool profiling_run = (argc > 1) ? (std::stoi(argv[1]) == 1) : false; 
        const unsigned int mesh_refinement_level = (argc > 2) ? std::stoi(argv[2]) : 4; 
        const bool enable_multigrid = (argc > 3) ? (std::strcmp(argv[3], "-mg") == 0) : false; 
        const unsigned int degree_finite_element = (argc > 4) ? std::stoi(argv[4]) : 2; 

        const unsigned int dimension = 2;

        const std::vector<double> beta_components = {1.0, 1.0};
        
        const auto mu      = std::make_shared<Functions::ConstantFunction<dimension, double>>(1.0);
        const auto beta    = std::make_shared<Functions::ConstantFunction<dimension, double>>(beta_components);
        const auto gamma   = std::make_shared<Functions::ConstantFunction<dimension, double>>(2.0);
        const auto forcing = std::make_shared<ForcingFunction<dimension>>();
        const auto h       = std::make_shared<Functions::ConstantFunction<dimension, double>>(0.0);
        const auto g       = std::make_shared<Functions::ConstantFunction<dimension, double>>(0.0);
        
        const std::set<types::boundary_id> dirichlet_ids = {};
        
        const std::set<types::boundary_id> neumann_ids = {0, 1, 2, 3};

        auto run_solver_for_degree = [&](auto degree_tag)
          {
            constexpr int p = decltype(degree_tag)::value;
    
            if(enable_multigrid) {
                MatrixFreeSolverMG<dimension, p, double> solver(
                    mu, beta, gamma, forcing, h, g, dirichlet_ids, neumann_ids, mesh_refinement_level
                );
                solver.run(profiling_run);
            } else {
                MatrixFreeSolver<dimension, p, double> solver(
                    mu, beta, gamma, forcing, h, g, dirichlet_ids, neumann_ids, mesh_refinement_level
                );
                solver.run(profiling_run);
            }
          };

          // runtime dispatch
          switch (degree_finite_element) {
              case 1: run_solver_for_degree(std::integral_constant<int, 1>{}); break;
              case 2: run_solver_for_degree(std::integral_constant<int, 2>{}); break;
              case 3: run_solver_for_degree(std::integral_constant<int, 3>{}); break;
              case 4: run_solver_for_degree(std::integral_constant<int, 4>{}); break;
              case 5: run_solver_for_degree(std::integral_constant<int, 5>{}); break;
              case 6: run_solver_for_degree(std::integral_constant<int, 6>{}); break;
              default:
                  pcout << "Unsupported FE degree for multigrid. Please choose between 1 and 6." << std::endl;
                  return 1;
          }
            return 0;
}