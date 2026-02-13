#include "DiffusionReaction-parallel.hpp"

void DiffusionReactionParallel::setup()
{
  pcout << "===============================================" << std::endl;

  // Create the mesh.
  {
    pcout << "Initializing the mesh" << std::endl;

    // Generiamo direttamente l'ipercubo distribuito, esattamente come nel Matrix-Free
    GridGenerator::hyper_cube(mesh, 0.0, 1.0, true);
    
    // NOTA: Devi passare la variabile mesh_refinement_level a questa funzione
    // oppure definirla nella classe del tuo amico.
    mesh.refine_global(mesh_refinement_level);

    pcout << "  Number of elements = " << mesh.n_global_active_cells()
          << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the finite element space. This is the same as in serial codes.
  {
    pcout << "Initializing the finite element space" << std::endl;

    fe = std::make_unique<FE_Q<dim>>(r); //MoDIFICA 6: Sostituito FE_SimplexP con FE_Q, che è più adatto per ipercubi

    pcout << "  Degree                     = " << fe->degree << std::endl;
    pcout << "  DoFs per cell              = " << fe->dofs_per_cell
          << std::endl;

    quadrature = std::make_unique<QGauss<dim>>(r + 1);
    quadrature_boundary = std::make_unique<QGauss<dim - 1>>(r + 1);

    pcout << "  Quadrature points per cell = " << quadrature->size()
          << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the DoF handler.
  {
    pcout << "Initializing the DoF handler" << std::endl;

    dof_handler.reinit(mesh);
    dof_handler.distribute_dofs(*fe);

    // We retrieve the set of locally owned DoFs, which will be useful when
    // initializing linear algebra classes.
    locally_owned_dofs = dof_handler.locally_owned_dofs();

    pcout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the linear system.
  {
    pcout << "Initializing the linear system" << std::endl;

    pcout << "  Initializing the sparsity pattern" << std::endl;

    // For the sparsity pattern, we use Trilinos' class, which manages some of
    // the inter-process communication.
    TrilinosWrappers::SparsityPattern sparsity(locally_owned_dofs,
                                               MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(dof_handler, sparsity);

    // After initialization, we need to call compress, so that all processes
    // retrieve the information they need from the rows they own (i.e. the rows
    // corresponding to locally owned DoFs).
    sparsity.compress();

    pcout << "  Initializing the system matrix" << std::endl;
    // Since the sparsity pattern is partitioned by row, so will be the matrix.
    system_matrix.reinit(sparsity);

    pcout << "  Initializing the system right-hand side" << std::endl;
    system_rhs.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    pcout << "  Initializing the solution vector" << std::endl;
    solution.reinit(locally_owned_dofs, MPI_COMM_WORLD);
  }
}

void DiffusionReactionParallel::assemble()
{
  pcout << "===============================================" << std::endl;

  pcout << "  Assembling the linear system" << std::endl;

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
      const double mu_loc = mu(fe_values.quadrature_point(q));
      const double sigma_loc = sigma(fe_values.quadrature_point(q));
      const double f_loc = f(fe_values.quadrature_point(q));
      const Tensor<1, dim> b_loc = b(fe_values.quadrature_point(q));

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        for (unsigned int j = 0; j < dofs_per_cell; ++j)
        {
          cell_matrix(i, j) += mu_loc *                     //
                               fe_values.shape_grad(i, q) * //
                               fe_values.shape_grad(j, q) * //
                               fe_values.JxW(q);

          cell_matrix(i, j) += sigma_loc *                   //
                               fe_values.shape_value(i, q) * //
                               fe_values.shape_value(j, q) * //
                               fe_values.JxW(q);
          cell_matrix(i, j) += (b_loc * fe_values.shape_grad(j, q)) * // b * grad(u)
                                fe_values.shape_value(i, q) * // v
                                fe_values.JxW(q);
        }

        cell_rhs(i) += f_loc *                       //
                       fe_values.shape_value(i, q) * //
                       fe_values.JxW(q);
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
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              cell_rhs(i) +=
                  phi(fe_face_values.quadrature_point(q)) * //
                  fe_face_values.shape_value(i, q) *        //
                  fe_face_values.JxW(q);
            }
          }
        }
      }
    }

    cell->get_dof_indices(dof_indices);

    system_matrix.add(dof_indices, cell_matrix);
    system_rhs.add(dof_indices, cell_rhs);
  }

  // Each process might have written to some rows it does not own (for instance,
  // if it owns elements that are adjacent to elements owned by some other
  // process). Therefore, at the end of assembly, processes need to exchange
  // information: the compress method allows to do this.
  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);

  // Dirichlet boundary conditions.
  {
    std::map<types::global_dof_index, double> boundary_values;
    FunctionG bc_function;

    std::map<types::boundary_id, const Function<dim> *> boundary_functions;
    boundary_functions[0] = &bc_function;
    boundary_functions[1] = &bc_function;
    boundary_functions[2] = &bc_function;
    boundary_functions[3] = &bc_function;

    VectorTools::interpolate_boundary_values(dof_handler,
                                             boundary_functions,
                                             boundary_values);

    MatrixTools::apply_boundary_values(
        boundary_values, system_matrix, solution, system_rhs, true);
  }
}

void DiffusionReactionParallel::solve()
{
  pcout << "===============================================" << std::endl;
  pcout << "Solving linear system..." << std::endl;

  Timer time; 
  unsigned int n_iter = 0;

  {
    // Precondizionatore Jacobi 
    TrilinosWrappers::PreconditionJacobi preconditioner;
    preconditioner.initialize(system_matrix);

    // Analysis of advection Term (Beta)
    bool is_advection_zero = true;
    for (unsigned int d = 0; d < dim; ++d)
    {
      if (std::abs(b(Point<dim>())[d]) > 1e-12)
      {
        is_advection_zero = false;
        break;
      }
    }

    // Tolleranza uniformata
    SolverControl solver_control(1000, 1e-12 * system_rhs.l2_norm());

    using VectorType = dealii::TrilinosWrappers::MPI::Vector;

    try
    {
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
    catch (const SolverControl::NoConvergence &e)
    {
      pcout << "Solver did not converge within "
            << solver_control.last_step() << std::endl;
      throw;
    }

    // Salviamo le iterazioni prima di uscire dallo scope
    n_iter = solver_control.last_step();
  } 

  // 3. Stesse Metriche di Performance del Matrix-Free
  const double elapsed_wall_time = time.wall_time();
  const double time_per_iter = (n_iter > 0) ? (elapsed_wall_time / n_iter) : 0.0;
  
  // Calcoliamo il throughput in MDoFs/s, tenendo conto del numero totale di DoFs e del tempo per iterazione
  const double throughput_mdofs = (time_per_iter > 1e-12) ? 
                                  (dof_handler.n_dofs() / time_per_iter / 1e6) : 0.0;

  pcout << "   Solved in " << n_iter << " iterations." << std::endl;
  pcout << "   Time per iter:      " << time_per_iter << " s" << std::endl;
  pcout << "   Throughput:         " << throughput_mdofs << " MDoFs/s" << std::endl;
  pcout << "===============================================" << std::endl;
}

void DiffusionReactionParallel::output() const
{
  pcout << "===============================================" << std::endl;
  pcout << "Outputting results..." << std::endl;

  // 1. Estrazione dei DoF "fantasma" (locally relevant)
  const IndexSet locally_relevant_dofs =
      DoFTools::extract_locally_relevant_dofs(dof_handler);

  // In Trilinos, i ghost si gestiscono creando un vettore ad hoc
  TrilinosWrappers::MPI::Vector solution_ghost(locally_owned_dofs,
                                               locally_relevant_dofs,
                                               MPI_COMM_WORLD);

  // L'assegnazione fa scattare la comunicazione MPI (scatter)
  solution_ghost = solution;

  // 2. Preparazione dell'output (Sintassi moderna uniformata)
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution_ghost, "solution");

  // Aggiungiamo il partizionamento MPI per la visualizzazione in Paraview
  std::vector<unsigned int> partition_int(mesh.n_active_cells());
  GridTools::get_subdomain_association(mesh, partition_int);
  const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  data_out.add_data_vector(partitioning, "partitioning");

  data_out.build_patches();

  // 3. Salvataggio su file
  // Rimosso mesh_file_name: usiamo un nome standard e il livello di raffinamento
  const std::string output_file_name = "solution_matrix_based";

  data_out.write_vtu_with_pvtu_record(
    "./",
    output_file_name,
    mesh_refinement_level, // Usiamo il livello di raffinamento come ciclo/indice
    MPI_COMM_WORLD,
    2 // Numero di cifre nel nome file (es. solution_matrix_based_03.vtu)
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
  // Usiamo la quadratura di Gauss standard (non Simplex)
  const QGauss<dim> quadrature_error(r + 2);

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