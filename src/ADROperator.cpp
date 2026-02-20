#include "ADROperator.hpp"

using namespace dealii;

/**
 * @file ADROperator.cpp
 * @brief Implementation of the ADROperator class for matrix-free solution of the Advection-Diffusion-Reaction equation.
 */

/**
 * @brief Constructor.
 * Initializes the base class with the correct vector type.
 */
template <int dim, int fe_degree, std::floating_point NumberType>
ADROperator<dim, fe_degree, NumberType>::ADROperator()
    : dealii::MatrixFreeOperators::Base<dim, dealii::LinearAlgebra::distributed::Vector<NumberType>>()
{}

/**
 * @brief Clears the operator state.
 * Resets the base class and clears the pre-computed coefficient tables to free memory.
 */
template <int dim, int fe_degree, std::floating_point NumberType>
void ADROperator<dim, fe_degree, NumberType>::clear()
{
    dealii::MatrixFreeOperators::Base<dim, dealii::LinearAlgebra::distributed::Vector<NumberType>>::clear();
    mu.reinit(0, 0);
    beta.reinit(0, 0);
    gamma_eff.reinit(0, 0);
}

/**
 * @brief Pre-computes PDE coefficients at quadrature points.
 *
 * It iterates over all cells (batches), evaluates the functions \f$ \mu, \beta, \gamma \f$
 * at the quadrature points, and stores them in SIMD-optimized tables.
 *
 * @note This method handles the SIMD "lane" unrolling manually to fill the `VectorizedArray`.
 */
template <int dim, int fe_degree, std::floating_point NumberType>
void ADROperator<dim, fe_degree, NumberType>::evaluate_coefficients(
    const Function<dim, NumberType> &mu_function,
    const Function<dim, NumberType> &beta_function,
    const Function<dim, NumberType> &gamma_function)
{
    // Initialize coefficient tables based on the mesh layout handled by MatrixFree
    const unsigned int n_cells = this->data->n_cell_batches();

    // Initialize FEEvaluation to query quadrature point info
    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, NumberType> phi(*this->data);
    const unsigned int n_q_points = phi.n_q_points;

    // Resize tables: [cell_batch_index][quadrature_point_index]
    mu.reinit(n_cells, n_q_points);
    beta.reinit(n_cells, n_q_points);
    gamma_eff.reinit(n_cells, n_q_points);

    for (unsigned int cell = 0; cell < n_cells; ++cell)
    {
        // Reinit FEEvaluation for the current batch to get correct physical points
        phi.reinit(cell);

        // Iterate over quadrature points
        for (unsigned int q = 0; q < phi.n_q_points; ++q)
        {
            // Get the point containing all vectorized quadrature points for this batch
            const auto vectorized_point = phi.quadrature_point(q);

            // Unroll over the SIMD lanes (e.g., 4 doubles for AVX256)
            for (unsigned int lane = 0; lane < VectorizedArray<NumberType>::size(); ++lane)
            {
                // 1. Extract the physical point for the current lane
                Point<dim, NumberType> qp;
                for (unsigned int d = 0; d < dim; ++d)
                    qp[d] = vectorized_point[d][lane];

                // 2. Evaluate analytic functions at this scalar point
                const NumberType mu_value = mu_function.value(qp);
                const NumberType gamma_value = gamma_function.value(qp);
                
                Tensor<1, dim, NumberType> beta_value;
                for (unsigned int d = 0; d < dim; ++d)
                    beta_value[d] = beta_function.value(qp, d);

                // 3. Calculate divergence of beta: \nabla \cdot \beta
                // WARNING: beta_function must implement gradient() correctly!
                NumberType div_beta_value = 0.0;
                for (unsigned int d = 0; d < dim; ++d)
                    div_beta_value += beta_function.gradient(qp, d)[d];

                // 4. Fill the SIMD data structures
                mu(cell, q)[lane] = mu_value;
                // Effective reaction: \gamma_{eff} = \gamma + \nabla \cdot \beta
                gamma_eff(cell, q)[lane] = gamma_value + div_beta_value; 
                
                for (unsigned int d = 0; d < dim; ++d)
                {
                    beta(cell, q)[d][lane] = beta_value[d];
                }
            }
        }
    }
}

/**
 * @brief Local kernel for the Matrix-Vector product.
 *
 * Computes the integral over the cell batch:
 * \f[
 * \int_{\Omega_e} \nabla v \cdot (\mu \nabla u) + v (\beta \cdot \nabla u + \gamma_{eff} u) dx
 * \f]
 *
 * @param data MatrixFree data object.
 * @param dst Destination vector (local accumulator).
 * @param src Source vector (local values).
 * @param cell_range Range of cell batches to process.
 */
template <int dim, int fe_degree, std::floating_point NumberType>
void ADROperator<dim, fe_degree, NumberType>::local_apply(
    const MatrixFree<dim, NumberType>                    &data,
    LinearAlgebra::distributed::Vector<NumberType>       &dst,
    const LinearAlgebra::distributed::Vector<NumberType> &src,
    const std::pair<unsigned int, unsigned int>          &cell_range) const
{
    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, NumberType> phi(data);
    
    // Loop over the assigned batch range (threading handled by MatrixFree::loop)
    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
        phi.reinit(cell);
        phi.read_dof_values(src);
        
        // Evaluate value and gradient of the source function u at quadrature points
        phi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

        for (unsigned int q = 0; q < phi.n_q_points; ++q)
        {
            const auto u_value = phi.get_value(q);
            const auto u_grad  = phi.get_gradient(q);

            // Fetch pre-computed coefficients (SIMD types)
            const auto mu_val    = mu[cell][q];
            const auto beta_val  = beta[cell][q];
            const auto gamma_val = gamma_eff[cell][q];

            // Weak form assembly at quadrature point
            // Term for test function v: ( \beta \cdot \nabla u + \gamma_{eff} u )
            auto val_term = (beta_val * u_grad) + (gamma_val * u_value);
            
            // Term for test function gradient \nabla v: ( \mu \nabla u )
            auto grad_term = mu_val * u_grad;

            // Submit terms to be integrated (multiplied by JxW and summed)
            // corresponds to: \int (grad_term * \nabla v + val_term * v)
            phi.submit_gradient(grad_term, q);
            phi.submit_value(val_term, q);
        }

        // Integrate and sum into local DoFs
        phi.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
        phi.distribute_local_to_global(dst);
    }
}

/**
 * @brief Public interface for the Matrix-Vector multiplication.
 * calls the parallel loop over cells.
 */
template <int dim, int fe_degree, std::floating_point NumberType>
void ADROperator<dim, fe_degree, NumberType>::apply_add(
    LinearAlgebra::distributed::Vector<NumberType>       &dst,
    const LinearAlgebra::distributed::Vector<NumberType> &src) const
{
    // Executes local_apply in parallel using TBB/MPI
    this->data->cell_loop(&ADROperator::local_apply, this, dst, src);
}

/**
 * @brief Computes the inverse diagonal of the operator for preconditioning (Jacobi).
 *
 * This implementation approximates the diagonal by integrating:
 * \f$ \int \mu |\nabla \phi_i|^2 + \gamma_{eff} \phi_i^2 \f$
 * Note: The advective contribution to the diagonal is currently neglected/approximated.
 */
template <int dim, int fe_degree, std::floating_point NumberType>
void ADROperator<dim, fe_degree, NumberType>::compute_diagonal()
{
    this->inverse_diagonal_entries.reset(
        new DiagonalMatrix<LinearAlgebra::distributed::Vector<NumberType>>());
    LinearAlgebra::distributed::Vector<NumberType> &inverse_diagonal =
        this->inverse_diagonal_entries->get_vector();
    this->data->initialize_dof_vector(inverse_diagonal);

    // Compute the diagonal elements locally
    MatrixFreeTools::compute_diagonal(*this->data,
                                      inverse_diagonal,
                                      &ADROperator::local_compute_diagonal,
                                      this);
    
    // Set diagonal entries corresponding to Dirichlet constraints to 1.0
    // to avoid division by zero or bad scaling
    this->set_constrained_entries_to_one(inverse_diagonal);

    // Invert the diagonal entries to get the Jacobi preconditioner P = D^{-1}
    for (unsigned int i = 0; i < inverse_diagonal.locally_owned_size(); ++i)
    {
        NumberType diag_val = inverse_diagonal.local_element(i);
        
        // Safety check to avoid division by zero
        if (std::abs(diag_val) > std::numeric_limits<NumberType>::epsilon())
        {
            inverse_diagonal.local_element(i) = 1.0 / diag_val;
        }
        else
        {
            // Fallback for zero diagonal (should ideally not happen in well-posed ADR)
            inverse_diagonal.local_element(i) = 1.0; 
        }
    }
}

/**
 * @brief Local kernel for diagonal computation.
 * Calculates the diagonal entry for each DoF by testing the basis function against itself.
 */
template <int dim, int fe_degree, std::floating_point NumberType>
void ADROperator<dim, fe_degree, NumberType>::local_compute_diagonal(
    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, NumberType> &phi) const
{
    const unsigned int cell = phi.get_current_cell_index();
    phi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

    for (const unsigned int q : phi.quadrature_point_indices())
    {
        const auto mu_val    = mu[cell][q];
        const auto beta_val  = beta[cell][q];
        const auto gamma_val = gamma_eff[cell][q];

        auto grad_term = mu_val * phi.get_gradient(q);
        auto val_term  = (beta_val * phi.get_gradient(q)) + (gamma_val * phi.get_value(q));

        phi.submit_gradient(grad_term, q);
        phi.submit_value(val_term, q);
    }
    phi.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
}

// Explicit template instantiations
template class ADROperator<1, 1, double>;
template class ADROperator<1, 2, double>;
template class ADROperator<1, 3, double>;
template class ADROperator<1, 4, double>;
template class ADROperator<1, 5, double>;
template class ADROperator<1, 6, double>;
template class ADROperator<1, 7, double>;
template class ADROperator<1, 8, double>;
template class ADROperator<1, 9, double>;
template class ADROperator<1, 10, double>;
template class ADROperator<1, 11, double>;
template class ADROperator<1, 12, double>;

template class ADROperator<2, 1, double>;
template class ADROperator<2, 2, double>;
template class ADROperator<2, 3, double>;
template class ADROperator<2, 4, double>;
template class ADROperator<2, 5, double>;
template class ADROperator<2, 6, double>;
template class ADROperator<2, 7, double>;
template class ADROperator<2, 8, double>;
template class ADROperator<2, 9, double>;
template class ADROperator<2, 10, double>;
template class ADROperator<2, 11, double>;
template class ADROperator<2, 12, double>;


template class ADROperator<3, 1, double>;
template class ADROperator<3, 2, double>;
template class ADROperator<3, 3, double>;
template class ADROperator<3, 4, double>;
template class ADROperator<3, 5, double>;
template class ADROperator<3, 6, double>;
template class ADROperator<3, 7, double>;
template class ADROperator<3, 8, double>;
template class ADROperator<3, 9, double>;
template class ADROperator<3, 10, double>;
template class ADROperator<3, 11, double>;
template class ADROperator<3, 12, double>;