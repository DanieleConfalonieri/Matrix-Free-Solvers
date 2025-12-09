#ifndef MATRIX_FREE_SOLVER_HPP
#define MATRIX_FREE_SOLVER_HPP

#include <concepts>
#include <iostream>
#include <fstream>
#include <memory> 

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

template <int dim, int fe_degree, std::floating_point NumberType>
class MatrixFreeSolver
{
    public:
    MatrixFreeSolver(
        std::shared_ptr<const dealii::Function<dim, NumberType>> mu_func,
        std::shared_ptr<const dealii::Function<dim, NumberType>> beta_func,
        std::shared_ptr<const dealii::Function<dim, NumberType>> gamma_func
    );
    void run();

    private:
    void setup_system();
    void assemble_rhs();
    void solve();
    void output_results() const;

#ifdef DEAL_II_WITH_P4EST
    dealii::parallel::distributed::Triangulation<dim> triangulation{MPI_COMM_WORLD};
#else
    dealii::Triangulation<dim> triangulation;
#endif
    const dealii::FE_Q<dim> fe;
    dealii::DoFHandler<dim> dof_handler;
    const dealii::MappingQ1<dim> mapping;

    dealii::AffineConstraints<NumberType> constraints;

    std::shared_ptr<dealii::MatrixFree<dim, NumberType>> matrix_free;

    using SystemMatrixType = 
        ADROperator<dim, fe_degree, NumberType>;
    SystemMatrixType system_matrix;

    // Physics related members
    std::shared_ptr<const dealii::Function<dim, NumberType>> mu_function;
    std::shared_ptr<const dealii::Function<dim, NumberType>> beta_function;
    std::shared_ptr<const dealii::Function<dim, NumberType>> gamma_function;
    
    /*
    MG related members
    */

    dealii::LinearAlgebra::distributed::Vector<NumberType> solution;
    dealii::LinearAlgebra::distributed::Vector<NumberType> system_rhs;

    NumberType setup_time;
    dealii::ConditionalOStream pcout;
    dealii::ConditionalOStream time_details;
};

#endif