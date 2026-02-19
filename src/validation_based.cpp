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
  virtual double value(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
  {
    return std::sin(M_PI * p[0]) * std::sin(M_PI * p[1]) * std::sin(M_PI * p[2]);
  }

  virtual Tensor<1, dim> gradient(const Point<dim> &p, const unsigned int = 0) const override
  {
    Tensor<1, dim> grad;
    const double px = M_PI * p[0];
    const double py = M_PI * p[1];
    const double pz = M_PI * p[2];

    grad[0] = M_PI * std::cos(px) * std::sin(py) * std::sin(pz);
    grad[1] = M_PI * std::sin(px) * std::cos(py) * std::sin(pz);
    grad[2] = M_PI * std::sin(px) * std::sin(py) * std::cos(pz);

    return grad;
  }
};

template <int dim>
class ForcingFunction : public Function<dim>
{
public:
  virtual double value(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
  {
    const double x = p[0];
    const double y = p[1];
    const double z = p[2];
    
    const double sx = std::sin(M_PI * x);
    const double sy = std::sin(M_PI * y);
    const double sz = std::sin(M_PI * z);
    
    const double cx = std::cos(M_PI * x);
    const double cy = std::cos(M_PI * y);
    const double cz = std::cos(M_PI * z);

    const double u_val = sx * sy * sz;

    const double term_diff_react = (3.0 * M_PI * M_PI + 1.0) * u_val;

    const double term_adv_x = M_PI * cx * sy * sz;
    const double term_adv_y = M_PI * sx * cy * sz;
    const double term_adv_z = M_PI * sx * sy * cz;
    
    return term_diff_react + term_adv_x + term_adv_y + term_adv_z;
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
      std::make_shared<Functions::ConstantFunction<dim>>(0.0),             // h (Neumann)
      std::make_shared<Functions::ConstantFunction<dim>>(0.0),             // g (Dirichlet)
      enable_multigrid);

      problem.set_dirichlet_ids({0,1,2,3,4,5});
      problem.set_neumann_ids({});

      problem.setup();
      problem.assemble();
      problem.solve();

      // Output scritto solo per l'ultimo livello se richiesto
      const bool profiling = (level == end_level) ? profiling_run : true; 
      if(!profiling) {
        problem.output(exact_solution);
      }

      // Calcolo errori
      double error_L2 = problem.compute_error(VectorTools::L2_norm, *exact_solution);
      double error_H1 = problem.compute_error(VectorTools::H1_norm, *exact_solution);

      // h = 1/2^L (cubo unitario)
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