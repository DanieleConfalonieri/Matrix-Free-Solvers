#ifndef DIFFUSION_REACTION_HPP
#define DIFFUSION_REACTION_HPP

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <filesystem>
#include <fstream>
#include <iostream>

using namespace dealii;

/**
 * Class managing the differential problem.
 */
class DiffusionReaction
{
public:
  // Physical dimension (1D, 2D, 3D)
  static constexpr unsigned int dim = 2;

  // Dirichlet boundary function.
  //
  // This is implemented as a dealii::Function<dim>, instead of e.g. a lambda
  // function, because this allows to use dealii boundary utilities directly.
  class FunctionG : public Function<dim>
  {
  public:
    // Constructor.
    FunctionG()
    {
    }

    // Evaluation.
    virtual double
    value(const Point<dim> &p,
          const unsigned int /*component*/ = 0) const override
    {
      return std::sin(3.0 * M_PI * p[0]) + std::sin(2.0 * M_PI * p[1]);
    }
  };

  // Constructor.
  DiffusionReaction(const std::string &mesh_file_name_,
                    const unsigned int &r_,
                    const std::function<double(const Point<dim> &)> &mu_,
                    const std::function<Tensor<1, dim>(const Point<dim> &)> &b_,
                    const std::function<double(const Point<dim> &)> &sigma_,
                    const std::function<double(const Point<dim> &)> &f_,
                    const std::function<double(const Point<dim> &)> &phi_)
      : mesh_file_name(mesh_file_name_), r(r_), mu(mu_), b(b_), sigma(sigma_), f(f_), phi(phi_)
  {
  }

  // Initialization.
  void
  setup();

  // System assembly.
  void
  assemble();

  // System solution.
  void
  solve();

  // Output.
  void
  output() const;

  // Compute the error against a given exact solution.
  double
  compute_error(const VectorTools::NormType &norm_type,
                const Function<dim> &exact_solution) const;

protected:
  // Name of the mesh.
  const std::string mesh_file_name;

  // Polynomial degree.
  const unsigned int r;

  // Diffusion coefficient.
  std::function<double(const Point<dim> &)> mu;

  // Bterm.
  std::function<Tensor<1, dim>(const Point<dim> &)> b;

  // Reaction coefficient.
  std::function<double(const Point<dim> &)> sigma;

  // Forcing term.
  std::function<double(const Point<dim> &)> f;

  // Phi term.
  std::function<double(const Point<dim> &)> phi;

  // Triangulation.
  Triangulation<dim> mesh;

  // Finite element space.
  std::unique_ptr<FiniteElement<dim>> fe;

  // Quadrature formula.
  std::unique_ptr<Quadrature<dim>> quadrature;

  // Quadrature formula for boundary integrals.
  std::unique_ptr<Quadrature<dim - 1>> quadrature_boundary;

  // DoF handler.
  DoFHandler<dim> dof_handler;

  // Sparsity pattern.
  SparsityPattern sparsity_pattern;

  // System matrix.
  SparseMatrix<double> system_matrix;

  // System right-hand side.
  Vector<double> system_rhs;

  // System solution.
  Vector<double> solution;
};

#endif