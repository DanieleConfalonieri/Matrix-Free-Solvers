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


  const auto mu = [](const Point<dim> &/*p*/) { return 1.0;};
  const auto b = [](const Point<dim> &/*p*/) { return Tensor<1, dim>({-1.0, -1.0}); };
  const auto f     = [](const Point<dim>     &/*p*/) { return 0.0; };
  const auto sigma = [](const Point<dim> & /*p*/) { return 1.0; };
  const auto phi   = [](const Point<dim> &/*p*/) { return 1.0; };
  
  DiffusionReactionParallel problem(mesh_refinement_level, degree, mu, b, sigma, f, phi);

  problem.setup();
  problem.assemble();
  problem.solve();
  problem.output();

  return 0;
}