#include "ADROperator.hpp"

//Libraries for dealii

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

template <int dim, int fe_degree, std::floating_point T>
  ADROperator<dim, fe_degree, number>::ADROperator()
    : MatrixFreeOperators::Base<dim,
                                LinearAlgebra::distributed::Vector<number>>()
  {}


template <int dim, int fe_degree, typename number>
  void ADROperator<dim, fe_degree, number>::clear()
  {
    coefficient.reinit(0, 0);
    MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<number>>::
      clear();
  }


  template <int dim, int fe_degree, typename number>
  void ADROperator<dim, fe_degree, number>::evaluate_coefficient(
    const Coefficient<dim> &coefficient_function)
  {
    const unsigned int n_cells = this->data->n_cell_batches();
    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number> phi(*this->data);
 
    coefficient.reinit(n_cells, phi.n_q_points);
    for (unsigned int cell = 0; cell < n_cells; ++cell)
      {
        phi.reinit(cell);
        for (const unsigned int q : phi.quadrature_point_indices())
          coefficient(cell, q) =
            coefficient_function.value(phi.quadrature_point(q));
      }
  }
 
 
 
 
  template <int dim, int fe_degree, typename number>
  void ADROperator<dim, fe_degree, number>::local_apply(
    const MatrixFree<dim, number>                    &data,
    LinearAlgebra::distributed::Vector<number>       &dst,
    const LinearAlgebra::distributed::Vector<number> &src,
    const std::pair<unsigned int, unsigned int>      &cell_range) const
  {
    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number> phi(data);
 
    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        AssertDimension(coefficient.size(0), data.n_cell_batches());
        AssertDimension(coefficient.size(1), phi.n_q_points);
 
        phi.reinit(cell);
        phi.read_dof_values(src);
        phi.evaluate(EvaluationFlags::gradients);
        for (const unsigned int q : phi.quadrature_point_indices())
          phi.submit_gradient(coefficient(cell, q) * phi.get_gradient(q), q);
        phi.integrate(EvaluationFlags::gradients);
        phi.distribute_local_to_global(dst);
      }
  }
 
 
 
  template <int dim, int fe_degree, typename number>
  void ADROperator<dim, fe_degree, number>::apply_add(
    LinearAlgebra::distributed::Vector<number>       &dst,
    const LinearAlgebra::distributed::Vector<number> &src) const
  {
    this->data->cell_loop(&ADROperator::local_apply, this, dst, src);
  }
 
 
 
  template <int dim, int fe_degree, typename number>
  void ADROperator<dim, fe_degree, number>::compute_diagonal()
  {
    this->inverse_diagonal_entries.reset(
      new DiagonalMatrix<LinearAlgebra::distributed::Vector<number>>());
    LinearAlgebra::distributed::Vector<number> &inverse_diagonal =
      this->inverse_diagonal_entries->get_vector();
    this->data->initialize_dof_vector(inverse_diagonal);
 
    MatrixFreeTools::compute_diagonal(*this->data,
                                      inverse_diagonal,
                                      &ADROperator::local_compute_diagonal,
                                      this);
 
    this->set_constrained_entries_to_one(inverse_diagonal);
 
    for (unsigned int i = 0; i < inverse_diagonal.locally_owned_size(); ++i)
      {
        Assert(inverse_diagonal.local_element(i) > 0.,
               ExcMessage("No diagonal entry in a positive definite operator "
                          "should be zero"));
        inverse_diagonal.local_element(i) =
          1. / inverse_diagonal.local_element(i);
      }
  }
 
 
 
  template <int dim, int fe_degree, typename number>
  void ADROperator<dim, fe_degree, number>::local_compute_diagonal(
    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number> &phi) const
  {
    const unsigned int cell = phi.get_current_cell_index();
 
    phi.evaluate(EvaluationFlags::gradients);
 
    for (const unsigned int q : phi.quadrature_point_indices())
      {
        phi.submit_gradient(coefficient(cell, q) * phi.get_gradient(q), q);
      }
    phi.integrate(EvaluationFlags::gradients);
  }

