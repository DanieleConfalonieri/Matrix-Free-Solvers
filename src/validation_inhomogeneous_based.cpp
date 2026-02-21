#include "DiffusionReaction-parallel.hpp"
#include <deal.II/base/function.h>
#include <cmath>
#include <memory>
#include <deal.II/base/convergence_table.h>
#include <deal.II/numerics/vector_tools.h>
#include <fstream>

using namespace dealii;

template <int dim>
class ExactSolution : public Function<dim>
{
    public:
  virtual double value(const Point<dim> &p, const unsigned int = 0) const override {
    return std::exp(p[0] + p[1] + p[2]);
  }

  virtual Tensor<1, dim> gradient(const Point<dim> &p, const unsigned int = 0) const override {
    const double val = std::exp(p[0] + p[1] + p[2]);
    Tensor<1, dim> grad;
    grad[0] = val; grad[1] = val; grad[2] = val;
    return grad;
  }
};

template <int dim>
class ForcingFunction : public Function<dim>
{
    public:
  virtual double value(const Point<dim> &p, const unsigned int = 0) const override {
    return std::exp(p[0] + p[1] + p[2]);
  }
};

template <int dim>
class NeumannFunction : public Function<dim>
{
public:
  virtual double value(const Point<dim> &p, const unsigned int = 0) const override {
    const double u_val = std::exp(p[0] + p[1] + p[2]);
    Tensor<1, dim> grad;
    grad[0] = u_val; grad[1] = u_val; grad[2] = u_val;

    Tensor<1, dim> normal;
    const double tol = 1e-7;
    if      (std::abs(p[0] - 0.0) < tol) normal[0] = -1.0;
    else if (std::abs(p[0] - 1.0) < tol) normal[0] =  1.0;
    
    if      (std::abs(p[1] - 0.0) < tol) normal[1] = -1.0;
    else if (std::abs(p[1] - 1.0) < tol) normal[1] =  1.0;
    
    if      (std::abs(p[2] - 0.0) < tol) normal[2] = -1.0;
    else if (std::abs(p[2] - 1.0) < tol) normal[2] =  1.0;

    return 1.0 * (grad * normal); 
  }
};

int main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  int my_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

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

  // Validation -> exact solution 
  const auto exact_solution = std::make_shared<ExactSolution<dim>>();

  ConvergenceTable table;
  std::ofstream convergence_file;
  if(my_rank == 0){
    convergence_file.open("convergence.csv");
    convergence_file << "h,eL2,eH1\n";
  }

  unsigned int start_level = 2; 
  unsigned int end_level = mesh_refinement_level < start_level ? start_level : mesh_refinement_level;

  pcout << "Running EOC analysys for p=" << fe_degree << " and mesh refinement level = " << end_level << std::endl;

  for (unsigned int level = start_level; level <= end_level; ++level) 
  {
      pcout << " -> Running Level L=" << level << "..." << std::endl;
      
      const std::vector<double> beta_components = {1.0, 1.0, 1.0};

      DiffusionReactionParallel problem(
      level,
      fe_degree,
      std::make_shared<Functions::ConstantFunction<dim>>(1.0),             // mu
      std::make_shared<Functions::ConstantFunction<dim>>(beta_components), // beta
      std::make_shared<Functions::ConstantFunction<dim>>(1.0),             // gamma
      std::make_shared<ForcingFunction<dim>>(),                            // forcing (USIAMO LA CLASSE CUSTOM)
      std::make_shared<NeumannFunction<dim>>(),                            // h (Neumann)
      std::make_shared<ExactSolution<dim>>(),             // g (Dirichlet)
      enable_multigrid);

      problem.set_dirichlet_ids({0,2,4});
      problem.set_neumann_ids({1,3,5});

      problem.setup();
      problem.assemble();
      problem.solve();

      // Output written just for the last level if required
      const bool profiling = (level == end_level) ? profiling_run : true; 
      if(!profiling) {
        problem.output(exact_solution);
      }

      // Claculate errors
      double error_L2 = problem.compute_error(VectorTools::L2_norm, *exact_solution);
      double error_H1 = problem.compute_error(VectorTools::H1_norm, *exact_solution);

      // h = 1/2^L (unitary cube)
      const double h = 1.0 / std::pow(2.0, level);
      table.add_value("h", h);
      table.add_value("L2", error_L2);
      table.add_value("H1", error_H1);
      
      if (my_rank == 0) {
        convergence_file << h << "," << error_L2 << "," << error_H1 << "\n";
      }
  }

  if (my_rank == 0) {
    convergence_file.close();
    table.evaluate_all_convergence_rates(ConvergenceTable::reduction_rate_log2);
    table.set_scientific("L2", true);
    table.set_scientific("H1", true);

    pcout << "\n=== Convergence Table ===\n";
    table.write_text(std::cout);
    pcout << "=========================\n";
  }

  return 0;
}