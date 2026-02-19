#include "DiffusionReaction-parallel.hpp"
#include <deal.II/base/timer.h>
#include <iomanip>

void DiffusionReactionParallel::setup()
{
  pcout << "===============================================" << std::endl;
  pcout << "Initializing the mesh" << std::endl;
  Timer time;
  cumulative_time = 0.0;
  // Create the mesh.
  {
    // Generiamo direttamente l'ipercubo distribuito, esattamente come nel Matrix-Free
    GridGenerator::hyper_cube(mesh, 0.0, 1.0, true);
    
    mesh.refine_global(mesh_refinement_level);

    

  // Initialize the finite element space. This is the same as in serial codes.
  

    fe = std::make_unique<FE_Q<dim>>(fe_degree);

  

    quadrature = std::make_unique<QGauss<dim>>(fe_degree + 1);
    quadrature_boundary = std::make_unique<QGauss<dim - 1>>(fe_degree + 1);

  // Initialize the DoF handler.
  
    dof_handler.reinit(mesh);
    dof_handler.distribute_dofs(*fe);

    // We retrieve the set of locally owned DoFs, which will be useful when
    // initializing linear algebra classes.
    locally_owned_dofs = dof_handler.locally_owned_dofs();

    // Initialize the linear system.
    // For the sparsity pattern, we use Trilinos' class, which manages some of
    // the inter-process communication.
    TrilinosWrappers::SparsityPattern sparsity(locally_owned_dofs,
                                               MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(dof_handler, sparsity);

    // After initialization, we need to call compress, so that all processes
    // retrieve the information they need from the rows they own (i.e. the rows
    // corresponding to locally owned DoFs).
    sparsity.compress();

    // Since the sparsity pattern is partitioned by row, so will be the matrix.
    system_matrix.reinit(sparsity);

    system_rhs.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    solution.reinit(locally_owned_dofs, MPI_COMM_WORLD);
  }
  cumulative_time += time.wall_time();
  // Prints some details about the mesh and DoF distribution.
  pcout << "Setup system               (CPU/wall) " << time.cpu_time()
               << "s/" << time.wall_time() << 's' << std::endl;
  pcout << "  Number of elements = " << mesh.n_global_active_cells()
          << std::endl;
  pcout << "-----------------------------------------------" << std::endl;
  pcout << "  Degree                     = " << fe->degree << std::endl;
  pcout << "  DoFs per cell              = " << fe->dofs_per_cell
        << std::endl;
  pcout << "  Quadrature points per cell = " << quadrature->size()
          << std::endl;
  pcout << "-----------------------------------------------" << std::endl;
  pcout << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;  

  pcout << "-----------------------------------------------" << std::endl;
}

void DiffusionReactionParallel::assemble()
{
  pcout << "===============================================" << std::endl;
  pcout << "  Assembling the linear system" << std::endl;
  Timer time;
  {
    // Constraints setup
    IndexSet locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);
    AffineConstraints<double> constraints;
    constraints.clear();
    constraints.reinit(locally_relevant_dofs);

    if (!dirichlet_ids.empty() && g_func)
    {
      std::map<types::boundary_id, const Function<dim> *> boundary_functions;
      for (const auto &id : dirichlet_ids)
        boundary_functions[id] = g_func.get();

      VectorTools::interpolate_boundary_values(dof_handler,
                                              boundary_functions,
                                              constraints);
    }
    constraints.close();

    // Number of local DoFs for each element.
    const unsigned int dofs_per_cell = fe->dofs_per_cell;

    // Number of quadrature points for each element.
    const unsigned int n_q = quadrature->size();

    FEValues<dim> fe_values(*fe,
                            *quadrature,
                            update_values | update_gradients |
                                update_quadrature_points | update_JxW_values);

    // Since we need to compute integrals on the boundary for Neumann conditions,
    // we also need a FEValues object to compute quantities on boundary edges
    // (faces).
    FEFaceValues<dim> fe_face_values(*fe,
                                    *quadrature_boundary,
                                    update_values | update_gradients |
                                        update_quadrature_points |
                                        update_JxW_values);

    // `neumann_ids` is configured by the caller via `set_neumann_ids(...)`.

    // Local matrix and vector.
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

    // Reset the global matrix and vector, just in case.
    system_matrix = 0.0;
    system_rhs = 0.0;

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      // If current cell is not locally owned, we skip it.
      if (!cell->is_locally_owned())
        continue;

      // On all other cells (which are owned by current process) we perform
      // assembly as usual.

      fe_values.reinit(cell);

      cell_matrix = 0.0;
      cell_rhs = 0.0;

      for (unsigned int q = 0; q < n_q; ++q)
      {
        const Point<dim> qp = fe_values.quadrature_point(q);
        const double mu_loc = mu_func->value(qp);
        const double gamma_loc = gamma_func->value(qp);
        double div_beta_loc = 0.0;
        for (unsigned int d = 0; d < dim; ++d)
          div_beta_loc += beta_func->gradient(qp, d)[d];
        const double gamma_eff_loc = gamma_loc + div_beta_loc; // γ_eff = γ + ∇·β
        const double f_loc = forcing_func->value(qp);
        Tensor<1, dim> beta_loc;
        for (unsigned int d = 0; d < dim; ++d)
          beta_loc[d] = beta_func->value(qp, d);

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
          {
            // Diffusion: mu * grad(u)·grad(v)
            cell_matrix(i, j) += mu_loc * fe_values.shape_grad(i, q) *
                                fe_values.shape_grad(j, q) * fe_values.JxW(q);

            // Reaction: gamma_eff*u*v
            cell_matrix(i, j) += gamma_eff_loc * fe_values.shape_value(i, q) *
                                fe_values.shape_value(j, q) * fe_values.JxW(q);

            // Advection: beta*grad(u)*v
            cell_matrix(i, j) += (beta_loc * fe_values.shape_grad(j, q)) *
                                fe_values.shape_value(i, q) * fe_values.JxW(q);
          }

          // RHS: f v
          cell_rhs(i) += f_loc * fe_values.shape_value(i, q) * fe_values.JxW(q);
        }
      }
      if (cell->at_boundary())
      {
        // ...we loop over its edges (referred to as faces in the deal.II
        // jargon).
        for (unsigned int face_no=0; face_no < GeometryInfo<dim>::faces_per_cell; ++face_no)
        {
          // If current face lies on the boundary and its boundary_id is in
          // the set of Neumann boundaries, assemble the boundary integral.
          auto face = cell->face(face_no);
          if (face->at_boundary() && neumann_ids.count(face->boundary_id()))
          {
            fe_face_values.reinit(cell, face_no);

            for (unsigned int q = 0; q < quadrature_boundary->size(); ++q)
            {
              const Point<dim> qp = fe_face_values.quadrature_point(q);
              const double mu_loc = mu_func->value(qp);
              const double h_loc = h_func->value(qp);
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                cell_rhs(i) += mu_loc * h_loc * fe_face_values.shape_value(i, q) *
                              fe_face_values.JxW(q);
            }
          }
        }
      }

      cell->get_dof_indices(dof_indices);

      constraints.distribute_local_to_global(cell_matrix, cell_rhs, dof_indices,
                                        system_matrix, system_rhs);
    }

    // Each process might have written to some rows it does not own (for instance,
    // if it owns elements that are adjacent to elements owned by some other
    // process). Therefore, at the end of assembly, processes need to exchange
    // information: the compress method allows to do this.
    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
  }
  cumulative_time += time.wall_time();
  pcout << "Assembly complete          (CPU/wall) " << time.cpu_time()
        << "s/" << time.wall_time() << 's' << std::endl;
}

void DiffusionReactionParallel::solve()
{
  pcout << "===============================================" << std::endl;
  pcout << "Solving linear system..." << std::endl;

  Timer timer;
  unsigned int n_iter = 0;

  {
    // Analysis of advection Term (Beta)
    bool is_advection_zero = true;
    for (unsigned int d = 0; d < dim; ++d)
    {
      if (std::abs(beta_func->value(Point<dim>(), d)) > 1e-12)
      {
        is_advection_zero = false;
        break;
      }
    }

    // Tolerance aligned with matrix-free
    SolverControl solver_control(10000, 1e-12);
    pcout << "  Solver tolerance: " << solver_control.tolerance() << std::endl;
    using VectorType = dealii::TrilinosWrappers::MPI::Vector;

    try
    {
      if (enable_multigrid)
      {
        // SETUP AMG 
        TrilinosWrappers::PreconditionAMG preconditioner;
        TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
        amg_data.elliptic = is_advection_zero;
        
        // Initialize the AMG preconditioner with the system matrix and the additional data
        preconditioner.initialize(system_matrix, amg_data);

        if (is_advection_zero)
        {
          pcout << "   -> Symmetric problem: AMG preconditioner + CG solver" << std::endl;
          SolverCG<VectorType> solver(solver_control);
          solver.solve(system_matrix, solution, system_rhs, preconditioner);
        }
        else
        {
          pcout << "   -> Non-symmetric problem (advection): AMG preconditioner + GMRES solver" << std::endl;
          SolverGMRES<VectorType> solver(solver_control);
          solver.solve(system_matrix, solution, system_rhs, preconditioner);
        }
      }
      else
      {
        TrilinosWrappers::PreconditionJacobi preconditioner;
        preconditioner.initialize(system_matrix);

        if (is_advection_zero)
        {
          pcout << "   -> Symmetric problem: Jacobi preconditioner + CG solver" << std::endl;
          SolverCG<VectorType> solver(solver_control);
          solver.solve(system_matrix, solution, system_rhs, preconditioner);
        }
        else
        {
          pcout << "   -> Non-symmetric problem (advection): Jacobi preconditioner + GMRES solver" << std::endl;
          SolverGMRES<VectorType> solver(solver_control);
          solver.solve(system_matrix, solution, system_rhs, preconditioner);
        }
      }
    }
    catch (const SolverControl::NoConvergence &e)
    {
      pcout << "Solver did not converge within "
            << solver_control.last_step() << std::endl;
      throw;
    }

    n_iter = solver_control.last_step();

    IndexSet locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);
    AffineConstraints<double> constraints;
    constraints.clear();
    constraints.reinit(locally_relevant_dofs);
    
    if (!dirichlet_ids.empty() && g_func)
    {
      std::map<types::boundary_id, const Function<dim> *> boundary_functions;
      for (const auto &id : dirichlet_ids)
        boundary_functions[id] = g_func.get();

      VectorTools::interpolate_boundary_values(dof_handler,
                                               boundary_functions,
                                               constraints);
    }
    constraints.close();
    
    constraints.distribute(solution);
  }

  // Stesse Metriche di Performance del Matrix-Free
  const double elapsed_wall_time = timer.wall_time();
  const double time_per_iter = elapsed_wall_time / n_iter;
  
  // Calcoliamo il throughput in MDoFs/s, tenendo conto del numero totale di DoFs e del tempo per iterazione
  const double throughput_mdofs = dof_handler.n_dofs() / time_per_iter / 1e6;// MDoFs/s
  
  pcout << "   Solve linear system       (CPU/wall) " << timer.cpu_time() << " s/" << elapsed_wall_time << 's' << std::endl;
  pcout << "   Solved in " << n_iter << " iterations." << std::endl;
  pcout << "   Time per iter:      " << time_per_iter << " s" << std::endl;
  pcout << "   Throughput:         " << throughput_mdofs << " MDoFs/s" << std::endl;
  pcout << "===============================================" << std::endl;
}

void DiffusionReactionParallel::output(const std::shared_ptr<const Function<dim>> &exact_solution) const
{
  pcout << "===============================================" << std::endl;
  pcout << "Outputting results..." << std::endl;

  // Locally relevant DoFs for parallel output
  const IndexSet locally_relevant_dofs =
      DoFTools::extract_locally_relevant_dofs(dof_handler);

  // Ghost vector 
  TrilinosWrappers::MPI::Vector solution_ghost(locally_owned_dofs,
                                               locally_relevant_dofs,
                                               MPI_COMM_WORLD);
  solution_ghost = solution; // Solution scatter

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution_ghost, "solution");

  if (exact_solution)
  {
      TrilinosWrappers::MPI::Vector exact_distributed(locally_owned_dofs, MPI_COMM_WORLD);
      VectorTools::interpolate(dof_handler, *exact_solution, exact_distributed);

      TrilinosWrappers::MPI::Vector exact_ghost(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
      exact_ghost = exact_distributed;

      TrilinosWrappers::MPI::Vector error_ghost(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
      error_ghost = solution_ghost; 
      error_ghost -= exact_ghost;   

      data_out.add_data_vector(exact_ghost, "exact_solution");
      data_out.add_data_vector(error_ghost, "error_nodal");
  }

  std::vector<unsigned int> partition_int(mesh.n_active_cells());
  GridTools::get_subdomain_association(mesh, partition_int);
  const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  data_out.add_data_vector(partitioning, "partitioning");

  data_out.build_patches();

  const std::string output_file_name = "solution_matrix_based";

  data_out.write_vtu_with_pvtu_record(
    "./",
    output_file_name,
    mesh_refinement_level,
    MPI_COMM_WORLD,
    2
  );

  pcout << "Output written to " << output_file_name << "_" 
        << std::setfill('0') << std::setw(2) << mesh_refinement_level 
        << ".pvtu" << std::endl;
  pcout << "===============================================" << std::endl;
}

double
DiffusionReactionParallel::compute_error(
    const VectorTools::NormType &norm_type,
    const Function<dim> &exact_solution) const
{
  const QGauss<dim> quadrature_error(fe_degree + 2);

  // Usiamo la mappatura nativa per elementi tensoriali 
  MappingQ<dim> mapping(1);

  Vector<double> error_per_cell(mesh.n_active_cells());
  
  VectorTools::integrate_difference(mapping,
                                    dof_handler,
                                    solution,
                                    exact_solution,
                                    error_per_cell,
                                    quadrature_error,
                                    norm_type);

  const double error =
      VectorTools::compute_global_error(mesh, error_per_cell, norm_type);

  return error;
}