#include "MatrixFreeSolver.hpp"
#include <deal.II/lac/solver_cg.h>

using namespace dealii;

/**
 * @file MatrixFreeSolver.cpp
 * @brief Minimal implementation of the matrix-free ADR solver.
 */

/**
 * @brief Construct solver with problem coefficients and boundary ids.
 */
template <int dim, int fe_degree, std::floating_point NumberType>
MatrixFreeSolver<dim, fe_degree, NumberType>::MatrixFreeSolver(
      std::shared_ptr<const dealii::Function<dim, NumberType>> mu_func,
      std::shared_ptr<const dealii::Function<dim, NumberType>> beta_func,
      std::shared_ptr<const dealii::Function<dim, NumberType>> gamma_func,
      std::shared_ptr<const dealii::Function<dim, NumberType>> forcing_func,
      std::shared_ptr<const dealii::Function<dim, NumberType>> neumann_func,
      std::shared_ptr<const dealii::Function<dim, NumberType>> dirichlet_func,
      const std::set<dealii::types::boundary_id> &dirichlet_b_ids,
      const std::set<dealii::types::boundary_id> &neumann_b_ids,
      const unsigned int mesh_refinement_level
)
#ifdef DEAL_II_WITH_P4EST
  : triangulation(MPI_COMM_WORLD, 
                  Triangulation<dim>::limit_level_difference_at_vertices,
                  parallel::distributed::Triangulation<dim>::Settings::default_setting)
#else
  : triangulation(Triangulation<dim>::limit_level_difference_at_vertices)
#endif
  , fe(fe_degree)
  , dof_handler(triangulation)
  , mu_function(mu_func)
  , beta_function(beta_func)
  , gamma_function(gamma_func)
  , forcing_function(forcing_func)
  , neumann_function(neumann_func)
  , dirichlet_function(dirichlet_func)
  , dirichlet_ids(dirichlet_b_ids)
  , neumann_ids(neumann_b_ids)
  , mesh_refinement_level(mesh_refinement_level)
  , cumulative_time(0.0)
  , pcout(std::cout,
          (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0))
  , time_details(std::cout,
                 (true && Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)) // Change false to true for profiling
{}

/**
 * @brief Sets up the grid, DoFs, and MatrixFree structures.
 */
template <int dim, int fe_degree, std::floating_point NumberType>
void MatrixFreeSolver<dim, fe_degree, NumberType>::setup_system()
{
  pcout << "===============================================" << std::endl;
  pcout << "Initializing the mesh" << std::endl;
  Timer time;
  cumulative_time = 0.0;
  {
    // 1. Grid Generation
    GridGenerator::hyper_cube(triangulation, 0.0, 1.0, true);
    triangulation.refine_global(mesh_refinement_level);

    system_matrix.clear();

    // 2. DoF Distribution
    dof_handler.distribute_dofs(fe);

    // 3. Constraints (Dirichlet BCs)
    constraints.clear();
    constraints.reinit(DoFTools::extract_locally_relevant_dofs(dof_handler));
    
    // Apply Dirichlet boundary conditions
    std::map<types::global_dof_index, NumberType> boundary_values;
    std::map<types::boundary_id, const Function<dim, NumberType> *> boundary_functions;
    
    for (const auto &b_id : dirichlet_ids)
      boundary_functions[b_id] = dirichlet_function.get();
    
    VectorTools::interpolate_boundary_values(
      mapping,
      dof_handler,
      boundary_functions,
      constraints);
    constraints.close();
  
    // 4. MatrixFree Initialization
    typename MatrixFree<dim, NumberType>::AdditionalData additional_data;
    additional_data.tasks_parallel_scheme =
      MatrixFree<dim, NumberType>::AdditionalData::none;
    
    // We need values and gradients for the PDE, and JxW for integration
    additional_data.mapping_update_flags =
      (update_values | update_gradients | update_JxW_values | update_quadrature_points);
    
    this->matrix_free = std::make_shared<MatrixFree<dim, NumberType>>();

    this->matrix_free->reinit(mapping,
                              dof_handler,
                              constraints,
                              QGauss<1>(fe.degree + 1),
                              additional_data);

    system_matrix.initialize(this->matrix_free);
  
  // 5. Coefficient Evaluation (Pre-computation)
  system_matrix.evaluate_coefficients(*mu_function, *beta_function, *gamma_function);

  // Initialize vectors
  system_matrix.initialize_dof_vector(solution);
  system_matrix.initialize_dof_vector(system_rhs);

  }
  cumulative_time += time.wall_time();
  const QGauss<dim> quadrature(fe.degree + 1);

  // Prints
  time_details << "Setup matrix-free system   (CPU/wall) " << time.cpu_time() << "s/ " << time.wall_time() << std::endl;
  pcout << "  Number of elements = " << triangulation.n_global_active_cells()
        << std::endl;
  pcout << "-----------------------------------------------" << std::endl;
  pcout << "  Degree                     = " << fe.degree << std::endl;
  pcout << "  DoFs per cell              = " << fe.dofs_per_cell
        << std::endl;
  pcout << "  Quadrature points per cell = " << quadrature.size()
        << std::endl;
  pcout << "-----------------------------------------------" << std::endl;
  pcout << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;  
  pcout << "-----------------------------------------------" << std::endl;
}

/**
 * @brief Assembles the global Right-Hand Side (RHS) vector.
 */
template <int dim, int fe_degree, std::floating_point NumberType>
void MatrixFreeSolver<dim, fe_degree, NumberType>::assemble_rhs()
{
  pcout << "===============================================" << std::endl;  
  pcout << "Assembling right hand side..." <<  std::endl;
  Timer time;
  {
    system_rhs = 0;

    // Create the vector for the Dirichlet shift (i.e. the contribution of Dirichlet BCs to the RHS)
    dealii::LinearAlgebra::distributed::Vector<NumberType> u_dirichlet;
    system_matrix.initialize_dof_vector(u_dirichlet);
    u_dirichlet = 0.0;
    constraints.distribute(u_dirichlet);
    u_dirichlet.update_ghost_values(); 
    
    // Quadrature and FEValues setup
    const QGauss<dim> quadrature_formula(fe_degree + 1);
    const QGauss<dim-1> face_quadrature_formula(fe_degree + 1);

    FEValues<dim> fe_values (
      fe,
      quadrature_formula,
      update_values | update_gradients | update_quadrature_points | update_JxW_values
    );

    FEFaceValues<dim> fe_face_values (
      fe,
      face_quadrature_formula,
      update_values | update_quadrature_points | update_JxW_values | update_boundary_forms
    );

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature_formula.size();
    const unsigned int n_face_q_points = face_quadrature_formula.size();

    // Local RHS vector and DoF indices
    Vector<NumberType> cell_rhs (dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    // Buffers for function values
    std::vector<NumberType> forcing_values (n_q_points);
    std::vector<NumberType> neumann_values (n_face_q_points);
    std::vector<NumberType> mu_boundary_values (n_face_q_points); 

    // Buffers for Dirichlet contributions
    std::vector<NumberType> mu_values(n_q_points);
    std::vector<Tensor<1, dim, NumberType>> beta_values(n_q_points);
    std::vector<NumberType> gamma_values(n_q_points);
    Vector<NumberType> local_dirichlet(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned())
        {
          cell_rhs = 0;
          fe_values.reinit(cell);

          // Extract Dirichlet values at DoFs of current cell
          cell->get_dof_indices(local_dof_indices);
          bool needs_shift = false;
          for (unsigned int i=0; i<dofs_per_cell; ++i){
            local_dirichlet(i) = u_dirichlet(local_dof_indices[i]);
            if (std::abs(local_dirichlet(i)) > 1e-14)
              needs_shift = true;
          }

          // Get forcing function values at quadrature points
          forcing_function->value_list(fe_values.get_quadrature_points(), forcing_values);

          if(needs_shift){
            mu_function->value_list(fe_values.get_quadrature_points(), mu_values);
            gamma_function->value_list(fe_values.get_quadrature_points(), gamma_values);
            
            // Populating manually the beta function values at quadrature points (since it's a vector-valued function)
            for (unsigned int q = 0; q < n_q_points; ++q) {
              for (unsigned int d = 0; d < dim; ++d) {
                beta_values[q][d] = beta_function->value(fe_values.quadrature_point(q), d);
              }
            }
          }

          // Volume integral: \int f * v
          for (unsigned int q=0; q<n_q_points; ++q)
            {
              const NumberType jxw = fe_values.JxW(q);
              NumberType u_g = 0;
              Tensor<1, dim, NumberType> grad_u_g;

              if(needs_shift){
                for(unsigned int j=0; j<dofs_per_cell; ++j){
                  u_g += local_dirichlet(j) * fe_values.shape_value(j, q);
                  for (unsigned int d=0; d<dim; ++d)
                    grad_u_g[d] += local_dirichlet(j) * fe_values.shape_grad(j, q)[d];
                }
              }
              for (unsigned int i=0; i<dofs_per_cell; ++i)
                {
                  NumberType rhs_val = forcing_values[q] * fe_values.shape_value(i, q); // f* v

                  // Apply the shift
                  if(needs_shift){
                    NumberType shift = mu_values[q] * (grad_u_g * fe_values.shape_grad(i, q)) + // mu * grad(u_g) * grad(v)
                                       (beta_values[q] * grad_u_g) * fe_values.shape_value(i, q) + // beta * grad(u_g) * v
                                       gamma_values[q] * u_g * fe_values.shape_value(i, q); // gamma * u_g * v 
                    rhs_val -= shift;
                  }
                  cell_rhs(i) += rhs_val * jxw;
                }
            }

          // Neumann BCs: \int mu * h * v (since flux is defined as mu * du/dn)
          for (unsigned int face_no=0; face_no < GeometryInfo<dim>::faces_per_cell; ++face_no)
            {
              if (cell->at_boundary(face_no) &&
                  neumann_ids.contains(cell->face(face_no)->boundary_id())) 
                    {
                      fe_face_values.reinit(cell, face_no);
                      // Evaluate Neumann function (h) and diffusion (mu) at face quadrature points
                      neumann_function->value_list(fe_face_values.get_quadrature_points(), neumann_values);
                      mu_function->value_list(fe_face_values.get_quadrature_points(), mu_boundary_values);

                      for (unsigned int q=0; q<n_face_q_points; ++q)
                        {
                          const NumberType jxw = fe_face_values.JxW(q);
                          // The flux term is \mu \nabla u \cdot n = \mu h (assuming standard flux definition)
                          const NumberType neumann_term = mu_boundary_values[q] * neumann_values[q];
                          
                          for (unsigned int i=0; i<dofs_per_cell; ++i)
                            {
                              cell_rhs(i) += fe_face_values.shape_value(i, q) *
                                            neumann_term *
                                            jxw;
                            }
                        }
                    }
              }
              // Distribute local contribution to global vector
              cell->get_dof_indices(local_dof_indices);
              constraints.distribute_local_to_global(cell_rhs, local_dof_indices, system_rhs);
        }
      }

      // Compress vector (communicate ghost values in parallel)
      system_rhs.compress(VectorOperation::add);
  }
  cumulative_time += time.wall_time();
  time_details << "Assembly right hand side   (CPU/wall) " << time.cpu_time()
            << "s/" << time.wall_time() << 's' << std::endl;
  }

/**
 * @brief Solves the linear system checking for Advection.
 * * Logic:
 * - If Beta == 0 (Symmetric): Use CG Solver.
 * - If Beta != 0 (Non-Symmetric): Use GMRES Solver.
 * * In both cases, a simple Jacobi Preconditioner (Inverse Diagonal) is used.
 */
template <int dim, int fe_degree, std::floating_point NumberType>
void MatrixFreeSolver<dim, fe_degree, NumberType>::solve()
{
  pcout << "===============================================" << std::endl;
  pcout << "Solving linear system..." << std::endl;
  
  Timer time;
  unsigned int n_iter = 0; 
  {
    // Compute the diagonal approximation for preconditioning
    system_matrix.compute_diagonal();
    const auto &inverse_diagonal = system_matrix.get_matrix_diagonal_inverse();
    
    // Simple local struct to apply Jacobi preconditioning: dst = D^{-1} * src
    struct JacobiPreconditioner
    {
      const VectorType &inv_diagonal;
      void vmult(VectorType &dst,
                 const VectorType &src) const
      {
        dst.equ(1.0, src);
        dst.scale(inv_diagonal);
      }
    };

    JacobiPreconditioner jacobi_preconditioner{inverse_diagonal->get_vector()};

    // Advection detection: β can be non-constant (evaluated at domain center).
    Point<dim> sample_point;
    for (unsigned int d = 0; d < dim; ++d)
      sample_point[d] = 0.5;
    bool is_advection_zero = true;
    for (unsigned int d = 0; d < dim; ++d) {
      if (std::abs(beta_function->value(sample_point, d)) > 1e-12) {
        is_advection_zero = false;
        break;
      }
    }
    // Solver configuration
    SolverControl solver_control (10000, 1e-12);
    pcout << "  Solver tolerance: " << solver_control.tolerance() << std::endl;
    constraints.set_zero(solution);

    try {
      if (is_advection_zero)
      {
        pcout << "   -> Symmetric problem: Jacobi preconditioner + CG solver" << std::endl;
        SolverCG<VectorType> solver(solver_control);
        solver.solve(system_matrix, solution, system_rhs, jacobi_preconditioner);
      }
      else
      {
        pcout << "   -> Non-symmetric problem (advection): Jacobi preconditioner + GMRES solver" << std::endl;
        SolverGMRES<VectorType> solver(solver_control);
        solver.solve(system_matrix, solution, system_rhs, jacobi_preconditioner);
      }
    }
    catch (const SolverControl::NoConvergence &e)
    {
      pcout << "Solver did not converge within "
            << solver_control.last_step()
            << std::endl;
      throw;
    }
    n_iter = solver_control.last_step();

    // Distribute Dirichlet values to the solution vector
    constraints.distribute(solution);
  }

  const double wall_time = time.wall_time();
  const double cpu_time = time.cpu_time();
  cumulative_time += wall_time;
  
  // Performance Metrics
  const double time_per_iter = wall_time / n_iter;
  
  // Throughput (Weak Scaling)
  const double throughput_mdofs = dof_handler.n_dofs() / time_per_iter / 1e6; // MDoFs/s

  time_details << "   Solve linear system       (CPU/wall) " << cpu_time
               << " s/" << wall_time << 's' << std::endl;
               
  time_details << "   Solved in " << n_iter << " iterations." << std::endl;
  time_details << "   Time per iter:   " << time_per_iter << " s" << std::endl;
  time_details << "   Throughput:             " << throughput_mdofs << " MDoFs/s" << std::endl;  
}

/**
 * @brief Outputs the results to VTU files.
 */
template <int dim, int fe_degree, std::floating_point NumberType>
void MatrixFreeSolver<dim, fe_degree, NumberType>::output_results(const unsigned int cycle, 
                                                                  const std::shared_ptr<Function<dim, NumberType>> exact_solution)
{
  Timer time;
  pcout << "Outputting results with exact solution comparison..." << std::endl;
  {
    DataOut<dim> data_out;
    solution.update_ghost_values();
    data_out.attach_dof_handler(dof_handler);

    // Add the numerical solution to the output
    data_out.add_data_vector(solution, "numerical_solution");

    if (exact_solution) 
    {
        // Distributed vector to hold the exact solution at DoFs
        VectorType exact_solution_vector;
        system_matrix.initialize_dof_vector(exact_solution_vector);
        
        // Interpolate the exact solution at DoFs
        VectorTools::interpolate(dof_handler, *exact_solution, exact_solution_vector);
        exact_solution_vector.update_ghost_values();
        data_out.add_data_vector(exact_solution_vector, "exact_solution");

        // Compute the error vector (numerical - exact)
        VectorType error_vector;
        system_matrix.initialize_dof_vector(error_vector);
        error_vector = solution;
        error_vector -= exact_solution_vector; // Ora error_vector contiene (u_h - u_ex)
        error_vector.update_ghost_values();
        data_out.add_data_vector(error_vector, "error_nodal");
    }

    data_out.build_patches(mapping);

    data_out.write_vtu_with_pvtu_record(
      "./",
      "solution",
      cycle,
      MPI_COMM_WORLD,
      2
    );
  }
  cumulative_time += time.wall_time();
  time_details << "Output results (incl. exact) (CPU/wall) " << time.cpu_time()
               << "s/" << time.wall_time() << 's' << std::endl;
}

template <int dim, int fe_degree, std::floating_point NumberType>
double MatrixFreeSolver<dim, fe_degree, NumberType>::compute_error(const VectorTools::NormType &norm_type,const Function<dim, NumberType> &exact_solution) const
{ 
  auto &non_const_sol = const_cast<VectorType &>(solution);
  non_const_sol.update_ghost_values();

  const QGauss<dim> quadrature_error(fe.degree + 2); 
  Vector<double> error_per_cell(triangulation.n_active_cells());
  
  VectorTools::integrate_difference(mapping,
                                    dof_handler,
                                    solution,
                                    exact_solution,
                                    error_per_cell,
                                    quadrature_error,
                                    norm_type);

  const double error = VectorTools::compute_global_error(triangulation, error_per_cell, norm_type);

  return error;
}

/**
 * @brief High-level function to run the simulation.
 */
template <int dim, int fe_degree, std::floating_point NumberType>
void MatrixFreeSolver<dim, fe_degree, NumberType>::run(const bool profiling_run, const std::shared_ptr<Function<dim, NumberType>> exact_solution)
{
  pcout << "Running MatrixFreeSolver..." << std::endl;
  setup_system();
  assemble_rhs();
  solve();
  if(!profiling_run)
    output_results(0, exact_solution);
}

// Explicit template instantiations
template class MatrixFreeSolver<2, 1, double>;
template class MatrixFreeSolver<2, 2, double>;
template class MatrixFreeSolver<2, 3, double>;
template class MatrixFreeSolver<2, 4, double>;
template class MatrixFreeSolver<2, 5, double>;
template class MatrixFreeSolver<2, 6, double>;
template class MatrixFreeSolver<2, 7, double>;
template class MatrixFreeSolver<2, 8, double>;
template class MatrixFreeSolver<2, 9, double>;
template class MatrixFreeSolver<2, 10, double>;
template class MatrixFreeSolver<2, 11, double>;
template class MatrixFreeSolver<2, 12, double>;

template class MatrixFreeSolver<3, 1, double>;
template class MatrixFreeSolver<3, 2, double>;
template class MatrixFreeSolver<3, 3, double>;
template class MatrixFreeSolver<3, 4, double>;
template class MatrixFreeSolver<3, 5, double>;
template class MatrixFreeSolver<3, 6, double>;
template class MatrixFreeSolver<3, 7, double>;
template class MatrixFreeSolver<3, 8, double>;
template class MatrixFreeSolver<3, 9, double>;
template class MatrixFreeSolver<3, 10, double>;
template class MatrixFreeSolver<3, 11, double>;
template class MatrixFreeSolver<3, 12, double>;