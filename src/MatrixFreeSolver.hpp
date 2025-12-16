#ifndef MATRIX_FREE_SOLVER_HPP
#define MATRIX_FREE_SOLVER_HPP

#include <concepts>
#include <iostream>
#include <fstream>
#include <memory> 
#include <set>

#include "ADROperator.hpp"

/**
 * @file MatrixFreeSolver.hpp
 * @brief Declaration of the MatrixFreeSolver class for orchestrating the matrix-free ADR solution.
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

/**
 * @brief Main driver class for the Matrix-Free ADR Solver.
 *
 * This class orchestrates the entire simulation pipeline:
 * - Mesh management (distributed triangulation).
 * - System setup (DoF distribution, MatrixFree initialization).
 * - Assembly of the Right-Hand Side (RHS), including volume forcing and Neumann boundary terms.
 * - Solving the linear system using an iterative solver (GMRES) with optional Multigrid preconditioning.
 * - Output of results for visualization.
 *
 * @tparam dim Spatial dimension.
 * @tparam fe_degree Polynomial degree of the Finite Element method.
 * @tparam NumberType Floating point type (float/double).
 */
template <int dim, int fe_degree, std::floating_point NumberType>
class MatrixFreeSolver
{
public:
    /// Type alias for the distributed vector.
    using VectorType = dealii::LinearAlgebra::distributed::Vector<NumberType>;

    /**
     * @brief Constructor.
     * Sets up the problem physics and boundary conditions.
     *
     * @param mu_func Shared pointer to the diffusion coefficient function \f$ \mu \f$.
     * @param beta_func Shared pointer to the advection field function \f$ \beta \f$.
     * @param gamma_func Shared pointer to the reaction coefficient function \f$ \gamma \f$.
     * @param forcing_func Shared pointer to the source term \f$ f \f$.
     * @param neumann_func Shared pointer to the Neumann boundary value \f$ h \f$ where \f$ \nabla u \cdot n = h \f$.
     * @param dirichlet_func Shared pointer to the Dirichlet boundary value \f$ g \f$ where \f$ u = g \f$.
     * @param dirichlet_b_ids Set of boundary IDs where Dirichlet conditions are applied.
     * @param neumann_b_ids Set of boundary IDs where Neumann conditions are applied.
     */
    MatrixFreeSolver(
        std::shared_ptr<const dealii::Function<dim, NumberType>> mu_func,
        std::shared_ptr<const dealii::Function<dim, NumberType>> beta_func,
        std::shared_ptr<const dealii::Function<dim, NumberType>> gamma_func,
        std::shared_ptr<const dealii::Function<dim, NumberType>> forcing_func,
        std::shared_ptr<const dealii::Function<dim, NumberType>> neumann_func,
        std::shared_ptr<const dealii::Function<dim, NumberType>> dirichlet_func,
        const std::set<dealii::types::boundary_id> &dirichlet_b_ids,
        const std::set<dealii::types::boundary_id> &neumann_b_ids
    );

    /**
     * @brief Executes the simulation workflow.
     * Calls setup, assembly, solve, and output methods in sequence.
     */
    void run();

private:
    /**
     * @brief Sets up the grid, DoFs, constraints, and MatrixFree structures.
     * Initializes the `matrix_free` object which computes geometry data and caches it for
     * the operator evaluation.
     */
    void setup_system();

    /**
     * @brief Assembles the global Right-Hand Side (RHS) vector.
     *
     * Computes the linear form \f$ F(v) \f$:
     * \f[
     * F(v) = \int_\Omega f v \, dx + \int_{\Gamma_N} h v \, ds
     * \f]
     * This method handles both the volume source term and the surface integral for Neumann boundaries.
     */
    void assemble_rhs();

    /**
     * @brief Solves the linear system \f$ A u = F \f$.
     * Uses a GMRES solver, likely preconditioned with a Matrix-Free Geometric Multigrid (GMG)
     * or a Chebyshev smoother depending on configuration.
     */
    void solve();

    /**
     * @brief Outputs the solution in VTU/Vtu format for Paraview.
     * @param cycle Current cycle index (useful for adaptive refinement or time-stepping).
     */
    void output_results(const unsigned int cycle);

    // --- Mesh & FE Data ---
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

    /// The central object for matrix-free evaluations.
    std::shared_ptr<dealii::MatrixFree<dim, NumberType>> matrix_free;

    using SystemMatrixType = ADROperator<dim, fe_degree, NumberType>;
    SystemMatrixType system_matrix;

    // --- Physics Functions ---
    std::shared_ptr<const dealii::Function<dim, NumberType>> mu_function;
    std::shared_ptr<const dealii::Function<dim, NumberType>> beta_function;
    std::shared_ptr<const dealii::Function<dim, NumberType>> gamma_function;

    // --- RHS Functions ---
    std::shared_ptr<const dealii::Function<dim, NumberType>> forcing_function;
    std::shared_ptr<const dealii::Function<dim, NumberType>> neumann_function;
    std::shared_ptr<const dealii::Function<dim, NumberType>> dirichlet_function;
    
    // --- Boundary IDs ---
    std::set<dealii::types::boundary_id> dirichlet_ids;
    std::set<dealii::types::boundary_id> neumann_ids;

    /*
    MG related members (Placeholder for Multigrid objects)
    Example:
    MGConstrainedDoFs mg_constrained_dofs;
    MGTransferMatrixFree<dim, NumberType> mg_transfer;
    */

    // --- Vectors ---
    dealii::LinearAlgebra::distributed::Vector<NumberType> solution;
    dealii::LinearAlgebra::distributed::Vector<NumberType> system_rhs;

    // --- Utilities ---
    NumberType setup_time;
    dealii::ConditionalOStream pcout;       ///< Stream for parallel output (prints only on rank 0).
    dealii::ConditionalOStream time_details;///< Stream for timing details.
};

#endif