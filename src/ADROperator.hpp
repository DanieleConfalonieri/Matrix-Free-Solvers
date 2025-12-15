#ifndef ADR_OPERATOR_HPP
#define ADR_OPERATOR_HPP

#include <concepts>
#include <iostream>
#include <fstream>

/**
 * @file ADROperator.hpp
 * @brief Declaration of the ADROperator class for matrix-free solution of the Advection-Diffusion-Reaction equation.
 */

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/solver_cg.h>
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
 * @brief Matrix-free operator for the Advection-Diffusion-Reaction (ADR) equation.
 *
 * This class implements the matrix-vector multiplication operation for the discretized
 * weak form of the ADR equation without assembling the global system matrix.
 * It leverages `dealii::MatrixFree` and `dealii::FEEvaluation` for SIMD-vectorized
 * evaluations on modern CPU architectures.
 *
 * The strong form of the problem is:
 * \f[
 * -\nabla \cdot (\mu \nabla u) + \nabla \cdot (\beta u) + \gamma u = f \quad \text{in } \Omega
 * \f]
 *
 * Depending on the implementation of the coefficients, the advective term might be expanded as:
 * \f[
 * \nabla \cdot (\beta u) = \beta \cdot \nabla u + (\nabla \cdot \beta) u
 * \f]
 * grouping the zero-th order terms into an effective reaction coefficient:
 * \f[
 * \gamma_{eff} = \gamma + \nabla \cdot \beta
 * \f]
 *
 * @tparam dim Spatial dimension of the problem.
 * @tparam fe_degree Polynomial degree of the Finite Element.
 * @tparam NumberType Floating point type (usually double or float).
 */
template<int dim, int fe_degree, std::floating_point NumberType>
class ADROperator
    : public dealii::MatrixFreeOperators::
        Base<dim, dealii::LinearAlgebra::distributed::Vector<NumberType>>
{
public:
    /**
     * @brief Type alias for the vectorized array type used in FEEvaluation.
     */
    using VectorType = dealii::LinearAlgebra::distributed::Vector<NumberType>;

    /**
     * @brief Constructor.
     */
    ADROperator();

    /**
     * @brief Clears the internal data structures and coefficient tables.
     */
    void clear() override;

    /**
     * @brief Pre-evaluates and stores the PDE coefficients at quadrature points.
     *
     * This function populates the internal tables `mu`, `beta`, and `gamma_eff`.
     * Computing these values once allows for faster matrix-vector multiplications,
     * trading memory bandwidth for computational reduction during the solver phase.
     *
     * @param mu_function The diffusion coefficient function \f$ \mu(x) \f$.
     * @param beta_function The advection field function \f$ \beta(x) \f$.
     * @param gamma_function The reaction coefficient function \f$ \gamma(x) \f$.
     */
    void evaluate_coefficients(const dealii::Function<dim, NumberType> &mu_function, 
                               const dealii::Function<dim, NumberType> &beta_function, 
                               const dealii::Function<dim, NumberType> &gamma_function);

    /**
     * @brief Computes the diagonal of the operator matrix.
     *
     * This is required for Chebyshev smoothing in Multigrid or Jacobi preconditioning.
     * It approximates the diagonal by integrating the operator action on basis functions locally.
     */
    virtual void compute_diagonal() override;

private:
    /**
     * @brief Applies the operator to a source vector and adds the result to a destination vector.
     *
     * Implementation of the operation \f$ dst = dst + A \cdot src \f$.
     * This function calls the `dealii::MatrixFree::loop` to iterate over cells in parallel.
     *
     * @param dst The destination vector.
     * @param src The source vector.
     */
    virtual void apply_add(
      VectorType       &dst,
      const VectorType &src) const override;

    /**
     * @brief Local evaluation kernel to be executed on each cell (or batch of cells).
     *
     * Performs the gathering of DoFs, evaluation at quadrature points (gradients and values),
     * application of the PDE form, and integration back to DoFs.
     *
     * @param data The MatrixFree data object.
     * @param dst Destination vector (local access).
     * @param src Source vector (local access).
     * @param cell_range The range of cells to process in this batch (SIMD batch).
     */
    void local_apply(const dealii::MatrixFree<dim, NumberType>            &data,
                VectorType       &dst,
                const VectorType &src,
                const std::pair<unsigned int, unsigned int> &cell_range) const;

    /**
     * @brief Local kernel for computing the matrix diagonal.
     *
     * @param phi The FEEvaluation object initialized for the current batch.
     */
    void local_compute_diagonal(
      dealii::FEEvaluation<dim, fe_degree, fe_degree + 1, 1, NumberType> &phi
    ) const;

    /**
     * @brief Storage for scalar coefficients at quadrature points.
     *
     * Stored as a Table where indices correspond to [cell_batch_index][quadrature_point_index].
     * Using VectorizedArray ensures SIMD-aligned memory access.
     * `gamma_eff` stores \f$ \gamma + \nabla \cdot \beta \f$.
     */
    dealii::Table<2, dealii::VectorizedArray<NumberType>> mu, gamma_eff;

    /**
     * @brief Storage for the advection vector field at quadrature points.
     */
    dealii::Table<2, dealii::Tensor<1, dim, dealii::VectorizedArray<NumberType>>> beta;
};
#endif