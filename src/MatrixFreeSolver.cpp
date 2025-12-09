#include "MatrixFreeSolver.hpp"

using namespace dealii;
template <int dim, int fe_degree, std::floating_point NumberType>
  MatrixFreeSolver<dim, fe_degree, NumberType>::MatrixFreeSolver(
        std::shared_ptr<const dealii::Function<dim, NumberType>> mu_func,
        std::shared_ptr<const dealii::Function<dim, NumberType>> beta_func,
        std::shared_ptr<const dealii::Function<dim, NumberType>> gamma_func,
        std::shared_ptr<const dealii::Function<dim, NumberType>> forcing_func,
        std::shared_ptr<const dealii::Function<dim, NumberType>> neumann_func
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
    , setup_time(0.0)
    , pcout(std::cout,
            (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0))
    , time_details(std::cout,
                   (false && Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)) //remove false to enable time details (do we want it?)
  {}

  template <int dim, int fe_degree, std::floating_point NumberType>
  void MatrixFreeSolver<dim, fe_degree, NumberType>::setup_system()
  {
    Timer time;
    pcout << "Setting up system..." << std::endl;
    setup_time = 0.0;
    {
      system_matrix.clear();

      dof_handler.distribute_dofs(fe);

      pcout << "Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl;

      constraints.clear();
      constraints.reinit(DoFTools::extract_locally_relevant_dofs(dof_handler));
      VectorTools::interpolate_boundary_values(
        mapping, dof_handler, 0, Functions::ZeroFunction<dim, NumberType>(), constraints);
      constraints.close();
    }
    setup_time += time.wall_time();
    time_details << "Distribute DoFs & B.C.     (CPU/wall) " << time.cpu_time() << "s/ " << time.wall_time() << std::endl;
    time.restart();

    {
      {
        typename MatrixFree<dim, NumberType>::AdditionalData additional_data;
        additional_data.tasks_parallel_scheme =
          MatrixFree<dim, NumberType>::AdditionalData::none;
        additional_data.mapping_update_flags =
          (update_values | update_gradients | update_JxW_values | update_quadrature_points);
        this -> matrix_free = std::make_shared<MatrixFree<dim, NumberType>>();

        this -> matrix_free ->reinit(mapping,
                                  dof_handler,
                                  constraints,
                                  QGauss<1>(fe.degree + 1),
                                  additional_data);

        system_matrix.initialize(this -> matrix_free);
      }
      system_matrix.evaluate_coefficients(*mu_function, *beta_function, *gamma_function);

      system_matrix.initialize_dof_vector(solution);
      system_matrix.initialize_dof_vector(system_rhs);
    }
    setup_time += time.wall_time();
    time_details << "Setup matrix-free system   (CPU/wall) " << time.cpu_time() << "s/ " << time.wall_time() << std::endl;
    time.restart();
    {
      /*
       *  MG setup would go here
       */
    }
    setup_time += time.wall_time();
    time_details << "Setup matrix-free levels   (CPU/wall) " << time.cpu_time() << "s/ " << time.wall_time() << std::endl;
  }

  template <int dim, int fe_degree, std::floating_point NumberType>
  void MatrixFreeSolver<dim, fe_degree, NumberType>::assemble_rhs()
  {
    // Classical RHS assembly with FEEvaluation since it's performed only once
    Timer time;
    
    pcout << "Assembling right hand side..." <<  std::endl;
    {
      system_rhs = 0;
      // Quadrature and FEEvaluation setup
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

      // Local RHS vector
      Vector<NumberType> cell_rhs (dofs_per_cell);
      std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

      // Vector to hold function values at quadrature points
      std::vector<NumberType> forcing_values (n_q_points);
      std::vector<NumberType> neumann_values (n_face_q_points);
      std::vector<NumberType> mu_boundary_values (n_face_q_points); //mu is needed on boundary for Neumann BCs

      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          if (cell->is_locally_owned())
          {
            cell_rhs = 0;
            fe_values.reinit(cell);

            // Get forcing function values at quadrature points
            forcing_function->value_list(fe_values.get_quadrature_points(), forcing_values);

            // Volume integral: f*v
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

            // Neumann BCs: mu*h*v
            for (unsigned int face_no=0; face_no < GeometryInfo<dim>::faces_per_cell; ++face_no)
              {
                if (cell->at_boundary(face_no) &&
                    cell->face(face_no)->boundary_id() == neumann_boundary_id) 
                      {
                        fe_face_values.reinit(cell, face_no);
                        // Evaluate Neumann function and mu at face quadrature points
                        neumann_function->value_list(fe_face_values.get_quadrature_points(), neumann_values);
                        mu_function->value_list(fe_face_values.get_quadrature_points(), mu_boundary_values);

                        for (unsigned int q=0; q<n_face_q_points; ++q)
                          {
                            const NumberType jxw = fe_face_values.JxW(q);
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
                cell->get_dof_indices(local_dof_indices);
                constraints.distribute_local_to_global(cell_rhs, local_dof_indices, system_rhs);
          }
        }

            system_rhs.compress(VectorOperation::add);
    }
    setup_time += time.wall_time();
    time_details << "Assemble right hand side   (CPU/wall) " << time.cpu_time()
              << "s/" << time.wall_time() << 's' << std::endl;
    }

  template <int dim, int fe_degree, std::floating_point NumberType>
  void MatrixFreeSolver<dim, fe_degree, NumberType>::solve()
  {
    Timer time;
    pcout << "Solving linear system..." << std::endl;
    {
      system_matrix.compute_diagonal();

      const auto &inverse_diagonal = system_matrix.get_matrix_diagonal_inverse();
      // Preconditioner: dst = D^{-1} * src
      
      // local struct to handle GMRES preconditioning
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

      SolverControl solver_control (1000, 1e-12 * system_rhs.l2_norm());
      //TODO distinguish between symmetric and non-symmetric solvers based on operator properties
      SolverGMRES<VectorType> solver (solver_control);
      constraints.set_zero(solution);
      constraints.set_zero(system_rhs);

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
      // Distribute constraints after interior solve
      constraints.distribute(solution);
      // Stats
      pcout << "   Solved in " << solver_control.last_step()
            << " iterations." << std::endl;
    }
    setup_time += time.wall_time();
    time_details << "Solve linear system       (CPU/wall) " << time.cpu_time()
                 << "s/" << time.wall_time() << 's' << std::endl;
  }

  template <int dim, int fe_degree, std::floating_point NumberType>
  void MatrixFreeSolver<dim, fe_degree, NumberType>::output_results(const unsigned int cycle)
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
        "./",
        "solution",
        cycle,
        MPI_COMM_WORLD,
        2
      );
    }
    setup_time += time.wall_time();
    time_details << "Output results            (CPU/wall) " << time.cpu_time()
                 << "s/" << time.wall_time() << 's' << std::endl;
  }

  template <int dim, int fe_degree, std::floating_point NumberType>
  void MatrixFreeSolver<dim, fe_degree, NumberType>::run()
  {
    pcout << "Running MatrixFreeSolver..." << std::endl;
    setup_system();
    assemble_rhs();
    solve();
    output_results(0);
  }
  template class MatrixFreeSolver<2, 2, double>;