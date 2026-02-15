#include "MatrixFreeSolverMG.hpp"
#include <algorithm>

using namespace dealii;

/**
 * @file MatrixFreeSolverMG.cpp
 * @brief Implementation of the MatrixFreeSolver using MultiGrid as a Preconditioner, class for solving the Advection-Diffusion-Reaction equation.
 */

template <int dim, int fe_degree, std::floating_point NumberType>
MatrixFreeSolverMG<dim, fe_degree, NumberType>::MatrixFreeSolverMG(
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
                  parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy)
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

template <int dim, int fe_degree, std::floating_point NumberType>
void MatrixFreeSolverMG<dim, fe_degree, NumberType>::setup_system()
{
  Timer time;
  pcout << "Setting up system..." << std::endl;
  cumulative_time = 0.0;
  {
    // 1. Grid Generation
    GridGenerator::hyper_cube(triangulation, 0.0, 1.0, true);
    triangulation.refine_global(mesh_refinement_level); 

    system_matrix.clear();
    mg_matrices.clear_elements(); // Added for MG

    // 2. DoF Distribution
    dof_handler.distribute_dofs(fe);
    dof_handler.distribute_mg_dofs(); // Added for MG

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
  cumulative_time += time.wall_time();
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

  cumulative_time += time.wall_time();
  time_details << "Setup matrix-free system   (CPU/wall) " << time.cpu_time() << "s/ " << time.wall_time() << std::endl;
  time.restart();
  
  // Geometric multigrid setup
  {
    const unsigned int nlevels = triangulation.n_global_levels();
    mg_matrices.resize(0, nlevels - 1);

    mg_constrained_dofs.initialize(dof_handler);
    mg_constrained_dofs.make_zero_boundary_constraints(dof_handler, dirichlet_ids);

    for (unsigned int level = 0; level < nlevels; ++level)
      {
        // Initialize level constraints using locally relevant DoFs
        AffineConstraints<NumberType> level_constraints(
          DoFTools::extract_locally_relevant_level_dofs(dof_handler, level));
        
        for (const types::global_dof_index dof_index : mg_constrained_dofs.get_boundary_indices(level))
          level_constraints.add_line(dof_index);
        level_constraints.close();

        typename MatrixFree<dim, NumberType>::AdditionalData additional_data;
        additional_data.tasks_parallel_scheme =
          MatrixFree<dim, NumberType>::AdditionalData::none;
        additional_data.mapping_update_flags =
          (update_values | update_gradients | update_JxW_values | update_quadrature_points);
        additional_data.mg_level = level;
        
        std::shared_ptr<MatrixFree<dim, NumberType>> mg_mf_storage_level =
          std::make_shared<MatrixFree<dim, NumberType>>();
          
        mg_mf_storage_level->reinit(mapping,
                                    dof_handler,
                                    level_constraints,
                                    QGauss<1>(fe.degree + 1),
                                    additional_data);

        mg_matrices[level].initialize(mg_mf_storage_level, mg_constrained_dofs, level);
        // Evaluate coefficients for this level
        mg_matrices[level].evaluate_coefficients(*mu_function, *beta_function, *gamma_function);
      }
  }
  // ---------------------------------------------------------

  cumulative_time += time.wall_time();
  time_details << "Setup matrix-free levels   (CPU/wall) " << time.cpu_time() << "s/ " << time.wall_time() << std::endl;
}

template <int dim, int fe_degree, std::floating_point NumberType>
void MatrixFreeSolverMG<dim, fe_degree, NumberType>::assemble_rhs()
{
  Timer time;
  pcout << "Assembling right hand side..." <<  std::endl;
  {
    system_rhs = 0;

    // Dirichlet shift: contribution of known Dirichlet values to the RHS (same as MatrixFreeSolver)
    dealii::LinearAlgebra::distributed::Vector<NumberType> u_dirichlet;
    system_matrix.initialize_dof_vector(u_dirichlet);
    u_dirichlet = 0.0;
    constraints.distribute(u_dirichlet);
    u_dirichlet.update_ghost_values();

    const QGauss<dim> quadrature_formula(fe_degree + 1);
    const QGauss<dim-1> face_quadrature_formula(fe_degree + 1);

    FEValues<dim> fe_values (
      fe, quadrature_formula,
      update_values | update_gradients | update_quadrature_points | update_JxW_values
    );

    FEFaceValues<dim> fe_face_values (
      fe, face_quadrature_formula,
      update_values | update_quadrature_points | update_JxW_values | update_boundary_forms
    );

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature_formula.size();
    const unsigned int n_face_q_points = face_quadrature_formula.size();

    Vector<NumberType> cell_rhs (dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    std::vector<NumberType> forcing_values (n_q_points);
    std::vector<NumberType> neumann_values (n_face_q_points);
    std::vector<NumberType> mu_boundary_values (n_face_q_points);

    // Buffers for Dirichlet shift (A * u_g)
    std::vector<NumberType> mu_values(n_q_points);
    std::vector<Tensor<1, dim, NumberType>> beta_values(n_q_points);
    std::vector<NumberType> gamma_values(n_q_points);
    Vector<NumberType> local_dirichlet(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
      {
        cell_rhs = 0;
        fe_values.reinit(cell);

        cell->get_dof_indices(local_dof_indices);
        bool needs_shift = false;
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          local_dirichlet(i) = u_dirichlet(local_dof_indices[i]);
          if (std::abs(local_dirichlet(i)) > 1e-14)
            needs_shift = true;
        }

        forcing_function->value_list(fe_values.get_quadrature_points(), forcing_values);

        if (needs_shift)
        {
          mu_function->value_list(fe_values.get_quadrature_points(), mu_values);
          gamma_function->value_list(fe_values.get_quadrature_points(), gamma_values);
          for (unsigned int q = 0; q < n_q_points; ++q)
            for (unsigned int d = 0; d < dim; ++d)
              beta_values[q][d] = beta_function->value(fe_values.quadrature_point(q), d);
        }

        // Volume integral: \int f * v minus Dirichlet shift (A * u_g)
        for (unsigned int q = 0; q < n_q_points; ++q)
        {
          const NumberType jxw = fe_values.JxW(q);
          NumberType u_g = 0;
          Tensor<1, dim, NumberType> grad_u_g;
          if (needs_shift)
          {
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
            {
              u_g += local_dirichlet(j) * fe_values.shape_value(j, q);
              for (unsigned int d = 0; d < dim; ++d)
                grad_u_g[d] += local_dirichlet(j) * fe_values.shape_grad(j, q)[d];
            }
          }
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            NumberType rhs_val = forcing_values[q] * fe_values.shape_value(i, q);
            if (needs_shift)
            {
              const NumberType shift =
                mu_values[q] * (grad_u_g * fe_values.shape_grad(i, q)) +
                (beta_values[q] * grad_u_g) * fe_values.shape_value(i, q) +
                gamma_values[q] * u_g * fe_values.shape_value(i, q);
              rhs_val -= shift;
            }
            cell_rhs(i) += rhs_val * jxw;
          }
        }

        // Neumann BCs: \int mu * h * v
        for (unsigned int face_no = 0; face_no < GeometryInfo<dim>::faces_per_cell; ++face_no)
          if (cell->at_boundary(face_no) && neumann_ids.contains(cell->face(face_no)->boundary_id()))
          {
            fe_face_values.reinit(cell, face_no);
            neumann_function->value_list(fe_face_values.get_quadrature_points(), neumann_values);
            mu_function->value_list(fe_face_values.get_quadrature_points(), mu_boundary_values);
            for (unsigned int q = 0; q < n_face_q_points; ++q)
            {
              const NumberType jxw = fe_face_values.JxW(q);
              const NumberType neumann_term = mu_boundary_values[q] * neumann_values[q];
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                cell_rhs(i) += fe_face_values.shape_value(i, q) * neumann_term * jxw;
            }
          }

        constraints.distribute_local_to_global(cell_rhs, local_dof_indices, system_rhs);
      }
    system_rhs.compress(VectorOperation::add);
  }
  cumulative_time += time.wall_time();
  time_details << "Assemble right hand side   (CPU/wall) " << time.cpu_time()
            << "s/" << time.wall_time() << 's' << std::endl;
}

template <int dim, int fe_degree, std::floating_point NumberType>
void MatrixFreeSolverMG<dim, fe_degree, NumberType>::solve()
{
  Timer time;
  pcout << "Solving linear system..." << std::endl;
  {
    system_matrix.compute_diagonal();

    // Check if advection is zero
    bool is_advection_zero = true;
    for (unsigned int d = 0; d < dim; ++d) {
      if (std::abs(beta_function->value(Point<dim>(), d)) > 1e-12) {
        is_advection_zero = false;
        break;
      }
    }

    if (is_advection_zero)
    {
      pcout << "   -> Symmetric problem: Chebyshev preconditioner + CG solver" << std::endl;

      MGTransferMatrixFree<dim, NumberType> mg_transfer(mg_constrained_dofs);
      mg_transfer.build(dof_handler);

      using SmootherType = PreconditionChebyshev<LevelMatrixType, VectorType>;
      mg::SmootherRelaxation<SmootherType, VectorType> mg_smoother;
      
      MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
      smoother_data.resize(0, triangulation.n_global_levels() - 1);
      
      for (unsigned int level = 0; level < triangulation.n_global_levels(); ++level)
      {
          if (level > 0)
            {
              smoother_data[level].smoothing_range     = 15.;
              smoother_data[level].degree              = 5;
              smoother_data[level].eig_cg_n_iterations = 10;
            }
          else
            {
              smoother_data[0].smoothing_range = 1e-3;
              smoother_data[0].degree          = numbers::invalid_unsigned_int;
              // Cap eigenvalue estimation iterations (coarse level can have many DoFs)
              smoother_data[0].eig_cg_n_iterations =
                std::min(mg_matrices[0].m(), static_cast<unsigned int>(200));
            }
          mg_matrices[level].compute_diagonal();
          smoother_data[level].preconditioner = mg_matrices[level].get_matrix_diagonal_inverse();
        }
      mg_smoother.initialize(mg_matrices, smoother_data);

      MGCoarseGridApplySmoother<VectorType> mg_coarse;
      mg_coarse.initialize(mg_smoother);

      mg::Matrix<VectorType> mg_matrix(mg_matrices);

      MGLevelObject<MatrixFreeOperators::MGInterfaceOperator<LevelMatrixType>> mg_interface_matrices;
      mg_interface_matrices.resize(0, triangulation.n_global_levels() - 1);
      
      for (unsigned int level = 0; level < triangulation.n_global_levels(); ++level)
        mg_interface_matrices[level].initialize(mg_matrices[level]);
        
      mg::Matrix<VectorType> mg_interface(mg_interface_matrices);

      Multigrid<VectorType> mg(mg_matrix, mg_coarse, mg_transfer, mg_smoother, mg_smoother);
      mg.set_edge_matrices(mg_interface, mg_interface);

      PreconditionMG<dim, VectorType, MGTransferMatrixFree<dim, NumberType>>
        preconditioner(dof_handler, mg, mg_transfer);

      SolverControl solver_control(1000, 1e-12 * system_rhs.l2_norm());
      SolverCG<VectorType> solver(solver_control); // Use of CG
      
      constraints.set_zero(solution);
      
      try {
        solver.solve(system_matrix, solution, system_rhs, preconditioner); 
      }
      catch (const SolverControl::NoConvergence &e) {
        pcout << "Solver did not converge within " << solver_control.last_step() << std::endl;
        throw;
      }
      
      constraints.distribute(solution);
      pcout << "   Solved in " << solver_control.last_step() << " iterations." << std::endl;
    }
    else
    {
      pcout << "   -> Non-symmetric problem (advection): weighted Jacobi preconditioner + GMRES solver" << std::endl;

      MGTransferMatrixFree<dim, NumberType> mg_transfer(mg_constrained_dofs);
      mg_transfer.build(dof_handler);

      // Use Chebyshev object to emulate a damped Jacobi smoother
      using SmootherType = PreconditionChebyshev<LevelMatrixType, VectorType>;
      mg::SmootherRelaxation<SmootherType, VectorType> mg_smoother;
      
      MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
      smoother_data.resize(0, triangulation.n_global_levels() - 1);
      
      for (unsigned int level = 0; level < triangulation.n_global_levels(); ++level)
        {
          // Setup damped Jacobi (omega = 0.6)
          smoother_data[level].degree = 1; // Grade 1 = Simple linear iteration
          smoother_data[level].eig_cg_n_iterations = 0; // No eigenvalue estimation needed for fixed damping
          smoother_data[level].max_eigenvalue = 1.0 / 0.6; // Corrisponding to omega = 0.6
          
          mg_matrices[level].compute_diagonal();
          smoother_data[level].preconditioner = mg_matrices[level].get_matrix_diagonal_inverse();
        }
      mg_smoother.initialize(mg_matrices, smoother_data);

      // Solver for the coarse grid (Iterative GMRES)
      using CoarsePreconditionerType = DiagonalMatrix<VectorType>;
      auto coarse_preconditioner = mg_matrices[0].get_matrix_diagonal_inverse();
      
      ReductionControl coarse_grid_control(1000, 1e-12, 1e-2);
      SolverGMRES<VectorType> coarse_solver(coarse_grid_control);
      MGCoarseGridIterativeSolver<VectorType, SolverGMRES<VectorType>, LevelMatrixType, CoarsePreconditionerType> 
        mg_coarse(coarse_solver, mg_matrices[0], *coarse_preconditioner);

      mg::Matrix<VectorType> mg_matrix(mg_matrices);

      MGLevelObject<MatrixFreeOperators::MGInterfaceOperator<LevelMatrixType>> mg_interface_matrices;
      mg_interface_matrices.resize(0, triangulation.n_global_levels() - 1);
      
      for (unsigned int level = 0; level < triangulation.n_global_levels(); ++level)
        mg_interface_matrices[level].initialize(mg_matrices[level]);
        
      mg::Matrix<VectorType> mg_interface(mg_interface_matrices);

      Multigrid<VectorType> mg(mg_matrix, mg_coarse, mg_transfer, mg_smoother, mg_smoother);
      mg.set_edge_matrices(mg_interface, mg_interface);

      PreconditionMG<dim, VectorType, MGTransferMatrixFree<dim, NumberType>>
        preconditioner(dof_handler, mg, mg_transfer);

      SolverControl solver_control(1000, 1e-12 * system_rhs.l2_norm());
      SolverGMRES<VectorType> solver(solver_control); // Use of GMRES
      
      constraints.set_zero(solution);
      
      try {
        solver.solve(system_matrix, solution, system_rhs, preconditioner); 
      }
      catch (const SolverControl::NoConvergence &e) {
        pcout << "Solver did not converge within " << solver_control.last_step() << std::endl;
        throw;
      }
      
      constraints.distribute(solution);
      pcout << "   Solved in " << solver_control.last_step() << " iterations." << std::endl;
    }
  }
  cumulative_time += time.wall_time();
  time_details << "Solve linear system       (CPU/wall) " << time.cpu_time()
               << "s/" << time.wall_time() << 's' << std::endl;
}

template <int dim, int fe_degree, std::floating_point NumberType>
void MatrixFreeSolverMG<dim, fe_degree, NumberType>::output_results(const unsigned int cycle)
{
  Timer time;
  pcout << "Outputting results..." << std::endl;
  {
    DataOut<dim> data_out;
    solution.update_ghost_values();
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "solution");
    data_out.build_patches(mapping);

    DataOutBase::VtkFlags vtk_flags;
    vtk_flags.compression_level = DataOutBase::VtkFlags::best_speed;
    data_out.set_flags(vtk_flags);
    
    data_out.write_vtu_with_pvtu_record(
      "./", "solution", cycle, MPI_COMM_WORLD, 2
    );
  }
  cumulative_time += time.wall_time();
  time_details << "Output results            (CPU/wall) " << time.cpu_time()
               << "s/" << time.wall_time() << 's' << std::endl;
}

template <int dim, int fe_degree, std::floating_point NumberType>
void MatrixFreeSolverMG<dim, fe_degree, NumberType>::run(const bool profiling_run)
{
  pcout << "Running MatrixFreeSolverMG..." << std::endl;
  setup_system();
  assemble_rhs();
  solve();
  if (!profiling_run)
    output_results(0);
}

// Explicit template instantiations
template class MatrixFreeSolverMG<2, 2, double>;
template class MatrixFreeSolverMG<3, 2, double>;