#include "DiffusionReaction-parallel.hpp"
#include <deal.II/base/function.h>
#include <cmath>
#include <memory>

using namespace dealii;

template <int dim>
class ForcingFunction : public Function<dim>
{
public:
  virtual double value(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
  {
    const double x = p[0];
    const double y = p[1];
    const double z = p[2];
    
    const double term_diff = (2.0 * M_PI * M_PI - 1)  * std::sin(M_PI * x) * std::sin(M_PI * y) * std::exp(z);
    const double term_adv = M_PI * std::cos(M_PI * x) * std::sin(M_PI * y) * std::exp(z) + M_PI * std::sin(M_PI * x) * std::cos(M_PI * y) * std::exp(z) + std::sin(M_PI * x) * std::sin(M_PI * y) * std::exp(z);
    const double term_rect = std::sin(M_PI * x) * std::sin(M_PI * y) * std::exp(z);
    
    return term_diff + term_adv + term_rect;
  }
};

int main(int argc, char *argv[])
{
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
  const unsigned int fe_degree = (argc > 4) ? std::stoi(argv[4]) : 2; 

  constexpr unsigned int dim = DiffusionReactionParallel::dim;

  const std::vector<double> beta_components = {1.0, 1.0, 1.0};

  DiffusionReactionParallel problem(
      mesh_refinement_level,
      fe_degree,
      std::make_shared<Functions::ConstantFunction<dim>>(1.0),             // mu
      std::make_shared<Functions::ConstantFunction<dim>>(beta_components), // beta
      std::make_shared<Functions::ConstantFunction<dim>>(1.0),             // gamma
      std::make_shared<ForcingFunction<dim>>(),                            // forcing (USIAMO LA CLASSE CUSTOM)
      std::make_shared<Functions::ConstantFunction<dim>>(0.0),             // h (Neumann)
      std::make_shared<Functions::ConstantFunction<dim>>(0.0),             // g (Dirichlet)
      enable_multigrid);

  problem.set_dirichlet_ids({0,1,2,3,4,5});
  problem.set_neumann_ids({});

  problem.setup();
  problem.assemble();
  problem.solve();
  
  if(!profiling_run)
    problem.output();

  return 0;
}