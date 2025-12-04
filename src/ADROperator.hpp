#include <concepts>
#include <iostream>
#include <fstream>

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
 

template<int dim, int degree_finite_element, std::floating_point NumberType>
class ADROperator
    : public dealii::MatrixFreeOperators::
        Base<dim, LinearAlgebra::distributed::Vector<NumberType>>
{
    public:
    using VectorType = LinearAlgebra::distributed::Vector<NumberType>;

    ADROperator();
 
    void clear() override;
 
    void evaluate_coefficients(const dealii::Function<dim, NumberType> &mu_function, 
                               const dealii::Function<dim, NumberType> &beta_function, 
                               const dealii::Function<dim, NumberType> &gamma_function); //Table filling
 
    virtual void compute_diagonal() override;
 
  private:
    virtual void apply_add(
      VectorType       &dst,
      const VectorType &src) const override;
 
    void
    local_apply(const dealii::MatrixFree<dim, NumberType>                    &data,
                VectorType       &dst,
                const VectorType &src,
                const std::pair<unsigned int, unsigned int> &cell_range) const;
 
    void local_compute_diagonal(
      const dealii::MatrixFree<dim, NumberType>   &data,
      VectorType                                  &dst,
      const unsigned int                          &dummy,
      const std::pair<unsigned int, unsigned int> &cell_range) const;
 
    dealii::Table<2, dealii::VectorizedArray<NumberType>> mu, gamma_eff; //gamma_eff = gamma + div(beta)
    dealii::Table<2, dealii::Tensor<1, dim, dealii::VectorizedArray<NumberType>>> beta;
};