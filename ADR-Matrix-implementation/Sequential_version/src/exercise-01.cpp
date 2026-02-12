#include "DiffusionReaction.hpp"

// Main function.
int
main(int /*argc*/, char * /*argv*/[])
{
  constexpr unsigned int dim = DiffusionReaction::dim;

  const std::string  mesh_filename = "../mesh/mesh-square-h0.100000.msh";
  const unsigned int degree        = 2;

  const auto mu = [](const Point<dim> &/*p*/) { return 1.0;};
  const auto b = [](const Point<dim> &/*p*/) { return Tensor<1, dim>({-1.0, -1.0}); };
  const auto f     = [](const Point<dim>     &/*p*/) { return 0.0; };
  const auto sigma = [](const Point<dim> & /*p*/) { return 1.0; };
  const auto phi   = [](const Point<dim> &/*p*/) { return 1.0; };

  DiffusionReaction problem(mesh_filename, degree, mu, b, sigma, f, phi);

  problem.setup();
  problem.assemble();
  problem.solve();
  problem.output();

  return 0;
}