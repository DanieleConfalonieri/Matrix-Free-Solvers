#include "ADROperator.hpp" 

#include <deal.II/base/utilities.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/matrix_free/matrix_free.h>

using namespace dealii;

//simple main function to test ADROperator compilation and basic functionality
int main(int argc, char **argv)
{
  
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  // Problem parameters
  const int dim = 2;
  const int fe_degree = 1;
  using NumberType = double;

  std::cout << "Setting up test environment..." << std::endl;

  // Mesh creation and DoFHandler setup
  parallel::distributed::Triangulation<dim> triangulation(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(2); // Un po' di celle per non avere tabelle vuote

  FE_Q<dim> fe(fe_degree);
  DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  MappingQ1<dim> mapping;
  AffineConstraints<double> constraints;
  constraints.close(); // No constraints for this simple test

  // 4. Setup MatrixFree (The data container)
  // Note: we use shared_ptr because the operator expects shared ownership
  auto matrix_free_data = std::make_shared<MatrixFree<dim, NumberType>>();
  
  // Basic configuration for MatrixFree
  typename MatrixFree<dim, NumberType>::AdditionalData additional_data;
  additional_data.tasks_parallel_scheme = MatrixFree<dim, NumberType>::AdditionalData::none;

  matrix_free_data->reinit(mapping, dof_handler, constraints, QGauss<1>(fe_degree + 1), additional_data);

  std::cout << "MatrixFree initialized. Cells: " << triangulation.n_active_cells() << std::endl;

  ADROperator<dim, fe_degree, NumberType> adr_op;
  
  // This method is inherited from MatrixFreeOperators::Base
  adr_op.initialize(matrix_free_data); 

  std::cout << "Operator initialized. Testing evaluate_coefficient..." << std::endl;

  // Test evaluate_coefficient
  Functions::ConstantFunction<dim, NumberType> mu_func(1.0);
  Functions::ConstantFunction<dim, NumberType> gamma_func(1.0);
  
  // For vector beta (dim components)
  std::vector<double> beta_vec_val(dim, 1.0); // beta = [1.0, 1.0]
  Functions::ConstantFunction<dim, NumberType> beta_func(beta_vec_val);

  adr_op.evaluate_coefficient(mu_func, beta_func, gamma_func);

  std::cout << "evaluate_coefficient passed!" << std::endl;
  
  // Test compute_diagonal
  std::cout << "Testing compute_diagonal..." << std::endl;
  adr_op.compute_diagonal();
  std::cout << "compute_diagonal passed!" << std::endl;

  std::cout << "All tests passed successfully." << std::endl;

  return 0;
}