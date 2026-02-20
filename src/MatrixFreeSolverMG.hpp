#ifndef MATRIX_FREE_SOLVER_MG_HPP
#define MATRIX_FREE_SOLVER_MG_HPP

#include <concepts>
#include <iostream>
#include <fstream>
#include <memory>
#include <set>

#include "ADROperator.hpp"

/**
 * @file MatrixFreeSolverMG.hpp
 * @brief Matrix-free solver with Geometric Multigrid (GMG) preconditioner.
 *
 * This header declares the `MatrixFreeSolverMG` template class which is a
 * variant of the matrix-free ADR solver that configures and exposes
 * multigrid-specific data structures (per-level operators, constrained DoFs,
 * and MG transfer utilities). The implementation focuses on building the
 * MatrixFree operator and the multigrid level operators used as a preconditioner
 * for an iterative Krylov solver.
 */

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/numerics/vector_tools.h>


/**
 * @brief Matrix-free solver configured to use a geometric multigrid preconditioner.
 *
 * Template parameters:
 * - `dim` : spatial dimension.
 * - `fe_degree` : polynomial degree of the finite element space.
 * - `NumberType` : floating point type (float/double).
 *
 * The class mirrors the responsibilities of `MatrixFreeSolver` while adding
 * multigrid-specific members such as `mg_matrices` and `mg_constrained_dofs`.
 */
template <int dim, int fe_degree, std::floating_point NumberType>
class MatrixFreeSolverMG
{
public:
    /// Distributed vector type used by deal.II linear algebra wrappers.
    using VectorType = dealii::LinearAlgebra::distributed::Vector<NumberType>;

    /**
     * @brief Construct the MG-enabled matrix-free solver.
     *
     * @param mu_func Diffusion coefficient function (shared pointer).
     * @param beta_func Advection field (shared pointer).
     * @param gamma_func Reaction coefficient function (shared pointer).
     * @param forcing_func Volume forcing / source term (shared pointer).
     * @param neumann_func Neumann boundary value function (shared pointer).
     * @param dirichlet_func Dirichlet boundary value function (shared pointer).
     * @param dirichlet_b_ids Set of boundary IDs to apply Dirichlet conditions.
     * @param neumann_b_ids Set of boundary IDs to apply Neumann conditions.
     * @param mesh_refinement_level Global refinement level for the initial mesh.
     */
    MatrixFreeSolverMG(
        std::shared_ptr<const dealii::Function<dim, NumberType>> mu_func,
        std::shared_ptr<const dealii::Function<dim, NumberType>> beta_func,
        std::shared_ptr<const dealii::Function<dim, NumberType>> gamma_func,
        std::shared_ptr<const dealii::Function<dim, NumberType>> forcing_func,
        std::shared_ptr<const dealii::Function<dim, NumberType>> neumann_func,
        std::shared_ptr<const dealii::Function<dim, NumberType>> dirichlet_func,
        const std::set<dealii::types::boundary_id> &dirichlet_b_ids,
        const std::set<dealii::types::boundary_id> &neumann_b_ids,
        const unsigned int mesh_refinement_level = 4
    );

    /**
     * @brief Run the end-to-end simulation: setup, assemble, solve, output.
     * @param profiling_run If true, enables detailed timing output for profiling.
     * @param exact_solution Optional shared pointer to an exact solution function for validation and output.
     */
    void run(const bool profiling_run = false, const std::shared_ptr<dealii::Function<dim, NumberType>> exact_solution = nullptr);

    /**
     * @brief Computes the global L2 error between the numerical and exact solution.
     */
    double compute_error(const dealii::VectorTools::NormType &norm_type, const dealii::Function<dim, NumberType> &exact_solution) const;


private:
    /**
     * @brief Create the mesh, distribute DoFs, initialize MatrixFree and MG data.
     */
    void setup_system();

    /**
     * @brief Assemble the right-hand side vector (volume + Neumann contributions).
     */
    void assemble_rhs();

    /**
     * @brief Solve the linear system using GMRES preconditioned with MG.
     */
    void solve();

    /**
     * @brief Write solution output (VTU) for visualization.
     * @param cycle The current cycle index (useful for time-stepping/adaptivity).
     * @param exact_solution Optional pointer to an exact solution function for error computation and output.
     */
    void output_results(const unsigned int cycle, const std::shared_ptr<dealii::Function<dim, NumberType>> exact_solution = nullptr);

#ifdef DEAL_II_WITH_P4EST
    dealii::parallel::distributed::Triangulation<dim> triangulation{MPI_COMM_WORLD};
#else
    dealii::Triangulation<dim> triangulation;
#endif

    const dealii::FE_Q<dim> fe;
    dealii::DoFHandler<dim> dof_handler;
    const dealii::MappingQ1<dim> mapping;

    /// Constraints for hanging nodes and Dirichlet boundary conditions.
    dealii::AffineConstraints<NumberType> constraints;

    /// MatrixFree container used for operator evaluation.
    std::shared_ptr<dealii::MatrixFree<dim, NumberType>> matrix_free;

    using SystemMatrixType = ADROperator<dim, fe_degree, NumberType>;
    /// The matrix-free realization of the ADR operator acting on vectors.
    SystemMatrixType system_matrix;

    // Physics functions (coefficients and forcing).
    std::shared_ptr<const dealii::Function<dim, NumberType>> mu_function;
    std::shared_ptr<const dealii::Function<dim, NumberType>> beta_function;
    std::shared_ptr<const dealii::Function<dim, NumberType>> gamma_function;

    std::shared_ptr<const dealii::Function<dim, NumberType>> forcing_function;
    std::shared_ptr<const dealii::Function<dim, NumberType>> neumann_function;
    std::shared_ptr<const dealii::Function<dim, NumberType>> dirichlet_function;
    
    // Boundary ID sets for applying boundary conditions.
    std::set<dealii::types::boundary_id> dirichlet_ids;
    std::set<dealii::types::boundary_id> neumann_ids;

    /// MG-constrained DoFs helper (keeps track of constraints across levels).
    dealii::MGConstrainedDoFs mg_constrained_dofs;
    
    using LevelMatrixType = ADROperator<dim, fe_degree, NumberType>;
    /**
     * @brief Per-level operator objects used by the MG preconditioner.
     *
     * Each entry holds a level-local view of the matrix-free ADR operator.
     */
    dealii::MGLevelObject<LevelMatrixType> mg_matrices;

    /// Solution and RHS vectors (distributed when running in parallel).
    dealii::LinearAlgebra::distributed::Vector<NumberType> solution;
    dealii::LinearAlgebra::distributed::Vector<NumberType> system_rhs;

    /// Cumulative time across setup, assemble, solve, output (for reporting).
    NumberType cumulative_time;

    /// Conditional output streams (print only on the master MPI rank).
    dealii::ConditionalOStream pcout;
    dealii::ConditionalOStream time_details;

    /// Mesh refinement to determine the degree of coarseness
    const unsigned int mesh_refinement_level;
};

#endif