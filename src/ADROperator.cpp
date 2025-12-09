#include "ADROperator.hpp"

using namespace dealii;

template <int dim, int fe_degree, std::floating_point NumberType>
  ADROperator<dim, fe_degree, NumberType>::ADROperator()
    : dealii::MatrixFreeOperators::Base<dim,dealii::LinearAlgebra::distributed::Vector<NumberType>>()
  {}


template <int dim, int fe_degree, std::floating_point NumberType>
  void ADROperator<dim, fe_degree, NumberType>::clear()
  {
      dealii::MatrixFreeOperators::Base<dim, dealii::LinearAlgebra::distributed::Vector<NumberType>>::clear();
      mu.reinit(0, 0);
      beta.reinit(0, 0);
      gamma_eff.reinit(0, 0);
  }


  template <int dim, int fe_degree, std::floating_point NumberType>
  void ADROperator<dim, fe_degree, NumberType>::evaluate_coefficients(
    const Function<dim, NumberType> &mu_function,
    const Function<dim, NumberType> &beta_function,
    const Function<dim, NumberType> &gamma_function)
  {
    //Initialize coefficient tables
    const unsigned int n_cells = this->data->n_cell_batches();

    //Obtain quadrature points
    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, NumberType> phi(*this->data);
    const unsigned int n_q_points = phi.n_q_points;

    mu.reinit(n_cells, n_q_points);
    beta.reinit(n_cells, n_q_points);
    gamma_eff.reinit(n_cells, n_q_points);

    for (unsigned int cell = 0; cell < n_cells; ++cell)
      {
        //reinitiate for each cell
        phi.reinit(cell);

        //evaluate quadrature points
        for (unsigned int q = 0; q < phi.n_q_points; ++q)
          {
          // point containing all vectorized quadrature points
          const auto vectorized_point = phi.quadrature_point(q);

          // unroll over the vectorized point
          for (unsigned int lane = 0; lane < VectorizedArray<NumberType>::size(); ++lane)
            {
              //exctract the point in the lane
              Point<dim, NumberType> qp;
              for (unsigned int d = 0; d < dim; ++d) // different access for each dimension
                qp[d] = vectorized_point[d][lane];
              //evaluate coefficients
              const NumberType mu_value = mu_function.value(qp);
              const NumberType gamma_value = gamma_function.value(qp);
              Tensor<1, dim, NumberType> beta_value;
              // beta is a vector function
              for (unsigned int d = 0; d < dim; ++d)
                beta_value[d] = beta_function.value(qp,d);
              //pointwise evaluation of div(beta) (tensor divergence)
              NumberType div_beta_value = 0.0;
              for (unsigned int d = 0; d < dim; ++d)
                div_beta_value += beta_function.gradient(qp,d)[d];

              // fill tables
              mu(cell, q)[lane] = mu_value;
              gamma_eff(cell, q)[lane] = gamma_value + div_beta_value; //gamma_eff = gamma + div(beta)
              for (unsigned int d = 0; d < dim; ++d)
                {
                  beta(cell, q)[d][lane] = beta_value[d];
                }
            }
          }
       }
    }  
 
 
 
 
  template <int dim, int fe_degree, std::floating_point NumberType>
  void ADROperator<dim, fe_degree, NumberType>::local_apply(
    const MatrixFree<dim, NumberType>                    &data,
    LinearAlgebra::distributed::Vector<NumberType>       &dst,
    const LinearAlgebra::distributed::Vector<NumberType> &src,
    const std::pair<unsigned int, unsigned int>      &cell_range) const
  {
    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, NumberType> phi(data);
    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        phi.reinit(cell);
        phi.read_dof_values(src);
        phi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

        for (unsigned int q=0; q < phi.n_q_points; ++q){
          const auto u_value = phi.get_value(q);
          const auto u_grad = phi.get_gradient(q);

          const auto mu_val    = mu[cell][q];
          const auto beta_val  = beta[cell][q];
          const auto gamma_val = gamma_eff[cell][q];

          auto val_term = (beta_val * u_grad) + (gamma_val * u_value);
          auto grad_term = mu_val * u_grad;

          // diffusion term
          phi.submit_gradient( grad_term , q);
          phi.submit_value( val_term , q);
        }
        phi.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
        phi.distribute_local_to_global(dst);
      }
  }
 
  template <int dim, int fe_degree, std::floating_point NumberType>
  void ADROperator<dim, fe_degree, NumberType>::apply_add(
    LinearAlgebra::distributed::Vector<NumberType>       &dst,
    const LinearAlgebra::distributed::Vector<NumberType> &src) const
  {
    this->data->cell_loop(&ADROperator::local_apply, this, dst, src);
  }
 
 
  template <int dim, int fe_degree, std::floating_point NumberType>
  void ADROperator<dim, fe_degree, NumberType>::compute_diagonal()
  {
    this->inverse_diagonal_entries.reset(
      new DiagonalMatrix<LinearAlgebra::distributed::Vector<NumberType>>());
    LinearAlgebra::distributed::Vector<NumberType> &inverse_diagonal =
      this->inverse_diagonal_entries->get_vector();
    this->data->initialize_dof_vector(inverse_diagonal);
 
    MatrixFreeTools::compute_diagonal(*this->data,
                                      inverse_diagonal,
                                      &ADROperator::local_compute_diagonal,
                                      this);
    this->set_constrained_entries_to_one(inverse_diagonal);
 
    for (unsigned int i = 0; i < inverse_diagonal.locally_owned_size(); ++i)
      {
        /* IS OUR OPERATOR POSITIVE DEFINITE?
        Assert(inverse_diagonal.local_element(i) > 0.,
               ExcMessage("No diagonal entry in a positive definite operator "
                          "should be zero"));
        */ 
        // Invert the diagonal entries to get Jacobi preconditioner
        if(std::abs(inverse_diagonal.local_element(i)) > std::numeric_limits<NumberType>::epsilon())
        {
          inverse_diagonal.local_element(i) = 1. / inverse_diagonal.local_element(i);
        }
      }
  }
 
 
 
  template <int dim, int fe_degree, std::floating_point NumberType>
  void ADROperator<dim, fe_degree, NumberType>::local_compute_diagonal(
    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, NumberType> &phi) const
  {
    //TO DISCUSS -> shall we consider beta? advection operator might be skew-symmetric with small diagonal terms
    const unsigned int cell = phi.get_current_cell_index();
 
    phi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
 
    for (const unsigned int q : phi.quadrature_point_indices())
      {
        const auto mu_val    = mu[cell][q];
        const auto gamma_val = gamma_eff[cell][q];
 
        // diffusion term: mu*grad(phi)*grad(phi)
        phi.submit_gradient( mu_val * phi.get_gradient(q) , q);
        // reaction + advection term: gamma*phi*phi
        phi.submit_value( gamma_val * phi.get_value(q) , q);
        // Note: advection term contribution to diagonal is neglected for now
      }
    phi.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
  }
  // Explicit template instantiations
  template class ADROperator<2, 1, double>;
  template class ADROperator<2, 2, double>;