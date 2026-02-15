#ifndef DIFFUSION_REACTION_HPP
#define DIFFUSION_REACTION_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
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
#include <deal.II/base/timer.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>

using namespace dealii;

/**
 * Matrix-based parallel solver for the ADR problem (aligned with matrix-free).
 * All coefficients and BCs use shared_ptr<const Function<dim>> as in
 * MatrixFreeSolver / MatrixFreeSolverMG. β can be non-constant; ∇·β is
 * computed from beta_func->gradient() for γ_eff = γ + ∇·β.
 */
class DiffusionReactionParallel
{
public:
  static constexpr unsigned int dim = 3;

  /**
   * @param mesh_refinement_level Global refinement level for the unit hypercube.
   * @param fe_degree Polynomial degree of FE_Q.
   * @param mu_func Diffusion coefficient μ (scalar).
   * @param beta_func Advection field β (vector-valued; gradient used for ∇·β).
   * @param gamma_func Reaction coefficient γ.
   * @param forcing_func Source term f.
   * @param h_func Neumann data h (μ∇u·n = μh on Γ_N).
   * @param g_func Dirichlet data g (u = g on Γ_D); same generality as h_func.
   * @param enable_multigrid_ Use AMG preconditioner when true.
   */
  DiffusionReactionParallel(const unsigned int mesh_refinement_level_,
                            const unsigned int fe_degree_,
                            std::shared_ptr<const Function<dim>> mu_func,
                            std::shared_ptr<const Function<dim>> beta_func,
                            std::shared_ptr<const Function<dim>> gamma_func,
                            std::shared_ptr<const Function<dim>> forcing_func,
                            std::shared_ptr<const Function<dim>> h_func,
                            std::shared_ptr<const Function<dim>> g_func,
                            const bool enable_multigrid_ = false)
    : mesh_refinement_level(mesh_refinement_level_)
    , fe_degree(fe_degree_)
    , mu_func(mu_func)
    , beta_func(beta_func)
    , gamma_func(gamma_func)
    , forcing_func(forcing_func)
    , h_func(h_func)
    , g_func(g_func)
    , enable_multigrid(enable_multigrid_)
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
  const unsigned int mesh_refinement_level;
  const unsigned int fe_degree;

  // ADR coefficients and BCs (shared_ptr<Function<dim>>, aligned with matrix-free)
  std::shared_ptr<const Function<dim>> mu_func;
  std::shared_ptr<const Function<dim>> beta_func;
  std::shared_ptr<const Function<dim>> gamma_func;
  std::shared_ptr<const Function<dim>> forcing_func;
  std::shared_ptr<const Function<dim>> h_func;
  std::shared_ptr<const Function<dim>> g_func;

  const bool enable_multigrid;

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