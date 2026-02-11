#include "MatrixFreeSolver.hpp"

using namespace dealii;

/**
 * @file MatrixFreeSolver.cpp
 * @brief Implementation of the MatrixFreeSolver class for solving the Advection-Diffusion-Reaction equation.
 */

/**
 * @brief Constructor.
 *
 * Initializes the parallel triangulation, Finite Element system, and DoF handler.
 * It also stores shared pointers to the physical coefficient functions and boundary conditions.
 *
 * @param mu_func Diffusion coefficient.
 * @param beta_func Advection field.
 * @param gamma_func Reaction coefficient.
 * @param forcing_func Forcing term (RHS).
 * @param neumann_func Neumann boundary condition value.
 * @param dirichlet_func Dirichlet boundary condition value.
 * @param dirichlet_b_ids Set of boundary IDs for Dirichlet BCs.
 * @param neumann_b_ids Set of boundary IDs for Neumann BCs.
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
      const std::set<dealii::types::boundary_id> &neumann_b_ids
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
  , setup_time(0.0)
  , pcout(std::cout,
          (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0))
  , time_details(std::cout,
                 (false && Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)) // Change false to true for profiling
{}

/**
 * @brief Sets up the grid, DoFs, and MatrixFree structures.
 *
 * Steps performed:
 * 1. Generates a hypercube grid and refines it globally.
 * 2. Distributes Degrees of Freedom (DoFs).
 * 3. Computes AffineConstraints (Dirichlet BCs and hanging nodes).
 * 4. Initializes the `MatrixFree` object for vectorized evaluation.
 * 5. Pre-evaluates PDE coefficients (mu, beta, gamma) for the matrix-free operator.
 */
template <int dim, int fe_degree, std::floating_point NumberType>
void MatrixFreeSolver<dim, fe_degree, NumberType>::setup_system()
{
  Timer time;
  pcout << "Setting up system..." << std::endl;
  setup_time = 0.0;
  {
    // 1. Grid Generation
    GridGenerator::hyper_cube(triangulation, 0.0, 1.0, true);
    triangulation.refine_global(4); 

    system_matrix.clear();

    // 2. DoF Distribution
    dof_handler.distribute_dofs(fe);

    pcout << "Number of degrees of freedom: " << dof_handler.n_dofs()
          << std::endl;

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
  }
  setup_time += time.wall_time();
  time_details << "Distribute DoFs & B.C.     (CPU/wall) " << time.cpu_time() << "s/ " << time.wall_time() << std::endl;
  time.restart();

  {
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
  }
  
  // 5. Coefficient Evaluation (Pre-computation)
  system_matrix.evaluate_coefficients(*mu_function, *beta_function, *gamma_function);

  // Initialize vectors
  system_matrix.initialize_dof_vector(solution);
  system_matrix.initialize_dof_vector(system_rhs);

  setup_time += time.wall_time();
  time_details << "Setup matrix-free system   (CPU/wall) " << time.cpu_time() << "s/ " << time.wall_time() << std::endl;
  time.restart();
  {
    /*
     * MG setup would go here if multigrid is implemented.
     */
  }
  setup_time += time.wall_time();
  time_details << "Setup matrix-free levels   (CPU/wall) " << time.cpu_time() << "s/ " << time.wall_time() << std::endl;
}

/**
 * @brief Assembles the global Right-Hand Side (RHS) vector.
 *
 * Computes:
 * \f[
 * F_i = \int_\Omega f \phi_i \, dx + \int_{\Gamma_N} \mu h \phi_i \, ds
 * \f]
 *
 * Uses standard `FEValues` and `FEFaceValues` iteration since this operation is done only once
 * and is not the performance bottleneck.
 */
template <int dim, int fe_degree, std::floating_point NumberType>
void MatrixFreeSolver<dim, fe_degree, NumberType>::assemble_rhs()
{
  Timer time;
  
  pcout << "Assembling right hand side..." <<  std::endl;
  {
    system_rhs = 0;
    
    // Quadrature and FEValues setup
    const QGauss<dim> quadrature_formula(fe_degree + 1);
    const QGauss<dim-1> face_quadrature_formula(fe_degree + 1);

    FEValues<dim> fe_values (
      fe,
      quadrature_formula,
      update_values | update_quadrature_points | update_JxW_values
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
    std::vector<NumberType> mu_boundary_values (n_face_q_points); // mu is needed for flux: mu * grad u * n = mu * h

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned())
        {
          cell_rhs = 0;
          fe_values.reinit(cell);

          // Get forcing function values at quadrature points
          forcing_function->value_list(fe_values.get_quadrature_points(), forcing_values);

          // Volume integral: \int f * v
          for (unsigned int q=0; q<n_q_points; ++q)
            {
              const NumberType jxw = fe_values.JxW(q);
              for (unsigned int i=0; i<dofs_per_cell; ++i)
                {
                  cell_rhs(i) += fe_values.shape_value(i, q) *
                                forcing_values[q] *
                                jxw;
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
                          // Ensure consistency with the problem definition for "h"
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
  setup_time += time.wall_time();
  time_details << "Assemble right hand side   (CPU/wall) " << time.cpu_time()
            << "s/" << time.wall_time() << 's' << std::endl;
  }

/**
 * @brief Solves the linear system using GMRES and a Jacobi preconditioner.
 *
 * Uses `SolverGMRES` suitable for non-symmetric systems (due to advection).
 * The preconditioner approximates the inverse using the diagonal of the operator matrix.
 */
template <int dim, int fe_degree, std::floating_point NumberType>
void MatrixFreeSolver<dim, fe_degree, NumberType>::solve()
{
  Timer time;
  pcout << "Solving linear system..." << std::endl;
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

    // Solver Control: Max 1000 iters or reduction by 1e-12
    SolverControl solver_control (1000, 1e-12 * system_rhs.l2_norm());
    
    // Using GMRES as the system is generally non-symmetric
    SolverGMRES<VectorType> solver (solver_control);
    
    // Reset vectors (apply constraints to zero for the system solve)
    constraints.set_zero(solution);
    // Note: constraints are already applied to system_rhs during assembly/distribution
    // but ensure hanging node constraints are consistent on RHS if needed.
    
    try{
      solver.solve(system_matrix,
                   solution,
                   system_rhs,
                   jacobi_preconditioner);
    }
    catch (const SolverControl::NoConvergence &e)
    {
      pcout << "Solver did not converge within "
            << solver_control.last_step()
            << std::endl;
      throw;
    }
    
    // Distribute Dirichlet values to the solution vector
    constraints.distribute(solution);
    
    // Stats
    pcout << "   Solved in " << solver_control.last_step()
          << " iterations." << std::endl;
  }
  setup_time += time.wall_time();
  time_details << "Solve linear system       (CPU/wall) " << time.cpu_time()
               << "s/" << time.wall_time() << 's' << std::endl;
}

/**
 * @brief Outputs the results to VTU files.
 *
 * Can run in parallel using PVTU records. The output includes ghost values
 * to ensure correct visualization at partition boundaries.
 *
 * @param cycle Cycle number appended to the filename (e.g., solution_0.vtu).
 */
template <int dim, int fe_degree, std::floating_point NumberType>
void MatrixFreeSolver<dim, fe_degree, NumberType>::output_results(const unsigned int cycle)
{
  Timer time;
  pcout << "Outputting results..." << std::endl;
  {
    DataOut<dim> data_out;

    // Ensure ghost values are present for output
    solution.update_ghost_values();
    
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "solution");
    data_out.build_patches(mapping);

    // Set flags for efficient I/O
    DataOutBase::VtkFlags vtk_flags;
    //vtk_flags.compression_level = DataOutBase::VtkFlags::best_speed; commented out to avoid version issues
    data_out.set_flags(vtk_flags);
    
    data_out.write_vtu_with_pvtu_record(
      "./",
      "solution",
      cycle,
      MPI_COMM_WORLD,
      2 // Number of digits in filename
    );
  }
  setup_time += time.wall_time();
  time_details << "Output results            (CPU/wall) " << time.cpu_time()
               << "s/" << time.wall_time() << 's' << std::endl;
}

/**
 * @brief High-level function to run the simulation.
 */
template <int dim, int fe_degree, std::floating_point NumberType>
void MatrixFreeSolver<dim, fe_degree, NumberType>::run()
{
  pcout << "Running MatrixFreeSolver..." << std::endl;
  setup_system();
  assemble_rhs();
  solve();
  output_results(0);
}

// Explicit template instantiations
template class MatrixFreeSolver<2, 2, double>;
template class MatrixFreeSolver<3, 2, double>;