#ifndef DIFFUSION_REACTION_HPP
#define DIFFUSION_REACTION_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

// MODIFICA 1: Sostituito fully_distributed_tria con la standard distributed tria
// (necessaria per usare GridGenerator::hyper_cube in parallelo)
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

// MODIFICA 2: Inclusi gli elementi tensoriali (FE_Q) invece dei simpliciali
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>

using namespace dealii;

/**
 * Class managing the differential problem.
 */
class DiffusionReactionParallel
{
public:
  // Physical dimension (1D, 2D, 3D)
  static constexpr unsigned int dim = 2;
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
  // MODIFICA 3: Sostituita la stringa del file con il livello di raffinamento
  DiffusionReactionParallel(const unsigned int mesh_refinement_level_,
                    const unsigned int &r_,
                    const std::function<double(const Point<dim> &)> &mu_,
                    const std::function<Tensor<1, dim>(const Point<dim> &)> &b_,
                    const std::function<double(const Point<dim> &)> &sigma_,
                    const std::function<double(const Point<dim> &)> &f_,
                    const std::function<double(const Point<dim> &)> &phi_)
    : mesh_refinement_level(mesh_refinement_level_)
    , r(r_)
    , mu(mu_)
    , b(b_)
    , sigma(sigma_)
    , f(f_)
    , phi(phi_)
    , mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
    , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , mesh(MPI_COMM_WORLD)
    , pcout(std::cout, mpi_rank == 0)
  {}

  // Initialization.
  void
  setup();

  // Set Neumann boundary ids.
  void set_neumann_ids(const std::set<types::boundary_id> &ids)
  {
    neumann_ids = ids;
  }

  // Set Dirichlet boundary ids.
  void set_dirichlet_ids(const std::set<types::boundary_id> &ids)
  {
    dirichlet_ids = ids;
  }

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
                const Function<dim>         &exact_solution) const;

protected:
  // MODIFICA 4: Rimosso mesh_file_name, inserito mesh_refinement_level
  const unsigned int mesh_refinement_level;

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

  // Number of MPI processes.
  const unsigned int mpi_size;

  // Rank of the current MPI process.
  const unsigned int mpi_rank;

  // MODIFICA 5: Passaggio a parallel::distributed::Triangulation standard
  parallel::distributed::Triangulation<dim> mesh;

  // Finite element space.
  std::unique_ptr<FiniteElement<dim>> fe;

  // Quadrature formula.
  std::unique_ptr<Quadrature<dim>> quadrature;

  // Quadrature formula for boundary integrals.
  std::unique_ptr<Quadrature<dim - 1>> quadrature_boundary;

  // DoF handler.
  DoFHandler<dim> dof_handler;

  // System matrix.
  TrilinosWrappers::SparseMatrix system_matrix;

  // System right-hand side.
  TrilinosWrappers::MPI::Vector system_rhs;

  // System solution.
  TrilinosWrappers::MPI::Vector solution;

  // Output stream for process 0.
  ConditionalOStream pcout;

  // Locally owned DoFs for current process.
  IndexSet locally_owned_dofs;

  // Neumann boundary ids to be set by the caller.
  std::set<types::boundary_id> neumann_ids;
  // Dirichlet boundary ids to be set by the caller.
  std::set<types::boundary_id> dirichlet_ids;
};

#endif