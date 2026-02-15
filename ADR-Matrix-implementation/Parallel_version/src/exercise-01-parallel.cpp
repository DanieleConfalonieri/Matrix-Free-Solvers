#include "DiffusionReaction-parallel.hpp"
#include <deal.II/base/function.h>

int
main(int argc, char *argv[])
{
  constexpr unsigned int dim = DiffusionReactionParallel::dim;

  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const unsigned int mesh_refinement_level = 4;
  const unsigned int fe_degree = 2;
  const bool enable_multigrid = true;

  const std::vector<double> beta_components = {1.0, -1.0, 1.0};

  DiffusionReactionParallel problem(
      mesh_refinement_level,
      fe_degree,
      std::make_shared<Functions::ConstantFunction<dim>>(3.0),   // mu
      std::make_shared<Functions::ConstantFunction<dim>>(beta_components), // beta
      std::make_shared<Functions::ConstantFunction<dim>>(2.0),   // gamma
      std::make_shared<Functions::ConstantFunction<dim>>(1.0),  // forcing
      std::make_shared<Functions::ConstantFunction<dim>>(2.0),   // h (Neumann)
      std::make_shared<Functions::ConstantFunction<dim>>(7.0),   // g (Dirichlet)
      enable_multigrid);

  problem.set_dirichlet_ids({1, 2, 3, 4});
  problem.set_neumann_ids({0, 5});

  problem.setup();
  problem.assemble();
  problem.solve();
  problem.output();

  return 0;
}