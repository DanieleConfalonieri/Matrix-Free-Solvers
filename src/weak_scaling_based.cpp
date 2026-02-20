#include "DiffusionReaction-parallel.hpp"
#include <deal.II/base/function.h>

int
main(int argc, char *argv[])
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
      
  const bool profiling_run = (argc > 1) ? (std::stoi(argv[1]) == 1) : false; // Set to true to disable output file (.vtk) generation for profiling runs

  const unsigned int mesh_refinement_level = (argc > 2) ? std::stoi(argv[2]) : 4; // Adjust from command line for finer/coarser mesh

  const bool enable_multigrid = (argc > 3) ? (std::strcmp(argv[3], "-mg") == 0) : false; // Set to true to enable multigrid preconditioning if "-mg" flag is provided

  const unsigned int fe_degree = (argc > 4) ? std::stoi(argv[4]) : 2; // Polynomial degree of FE_Q

  constexpr unsigned int dim = DiffusionReactionParallel::dim;

  DiffusionReactionParallel problem(
      mesh_refinement_level,
      fe_degree,
      std::make_shared<Functions::ConstantFunction<dim>>(3.0),   // mu
      std::make_shared<Functions::ConstantFunction<dim>>(1.0), // beta
      std::make_shared<Functions::ConstantFunction<dim>>(2.0),   // gamma
      std::make_shared<Functions::ConstantFunction<dim>>(1.0),  // forcing
      std::make_shared<Functions::ConstantFunction<dim>>(2.0),   // h (Neumann)
      std::make_shared<Functions::ConstantFunction<dim>>(7.0),   // g (Dirichlet)
      enable_multigrid);

  problem.set_dirichlet_ids({1});
  problem.set_neumann_ids({0});

  problem.setup();
  problem.assemble();
  problem.solve();
  if(!profiling_run)
    problem.output();

  return 0;
}