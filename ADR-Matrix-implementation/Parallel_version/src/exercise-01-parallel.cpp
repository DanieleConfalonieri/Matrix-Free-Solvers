#include "DiffusionReaction-parallel.hpp"

// Main function.
int
main(int argc, char *argv[])
{
  constexpr unsigned int dim = DiffusionReactionParallel::dim;

  // This object calls MPI_Init when it is constructed and MPI_Finalize when it
  // is destroyed. It also initializes several other libraries bundled with
  // dealii (e.g. p4est, PETSc, ...).
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  //const std::string  mesh_filename = "../mesh/mesh-square-h0.100000.msh";
  const unsigned int mesh_refinement_level = 4; // MODIFICA: Sostituito il nome del file con un livello di raffinamento globale
  const unsigned int degree        = 2;
  const bool enable_multigrid = true; 


  const auto mu = [](const Point<dim> &/*p*/) { return 3.0;};
  const auto b = [](const Point<dim> &/*p*/) { return Tensor<1, dim>({1.00, -1.0, 1.0}); };
  const auto f     = [](const Point<dim>     &/*p*/) { return 1.0; };
  const auto sigma = [](const Point<dim> & /*p*/) { return 2.0; };
  const auto neumann_func   = [](const Point<dim> &/*p*/) { return 2.0; };
  
  DiffusionReactionParallel problem(mesh_refinement_level, degree, mu, b, sigma, f, neumann_func, enable_multigrid);

  // Specify Dirichlet boundary ids 
  std::set<types::boundary_id> dirichlet = {1, 2, 3, 4};
  problem.set_dirichlet_ids(dirichlet);

  // Specify Neumann boundary ids 
  std::set<types::boundary_id> neumann = {0,5};
  problem.set_neumann_ids(neumann);

  problem.setup();
  problem.assemble();
  problem.solve();
  problem.output();

  return 0;
}