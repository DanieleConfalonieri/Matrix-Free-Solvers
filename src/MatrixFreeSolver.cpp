#include "MatrixFreeSolver.hpp"

using namespace dealii;
template <int dim, int fe_degree, std::floating_point NumberType>
  MatrixFreeSolver<dim, fe_degree, NumberType>::MatrixFreeSolver(
        std::shared_ptr<const dealii::Function<dim, NumberType>> mu_func,
        std::shared_ptr<const dealii::Function<dim, NumberType>> beta_func,
        std::shared_ptr<const dealii::Function<dim, NumberType>> gamma_func
  )
#ifdef DEAL_II_WITH_P4EST
    : triangulation(MPI_COMM_WORLD, 
                    Triangulation<dim>::limit_level_difference_at_vertices,
                    parallel::distributed::Triangulation<dim>::Settings::default_setting)
#else
    : triangulation(Triangulation<dim>::limit_level_difference_at_vertices)
#endif
    :
    , fe(fe_degree)
    , dof_handler(triangulation)
    , mu_function(mu_func)
    , beta_function(beta_func)
    , gamma_function(gamma_func)
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
    setup_time = 0.0;
    {
      system_matrix.clear();

      dof_handler.distribute_dofs(fe);

      pcout << "Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl;

      constraints.clear();
      constraints.reinit(dof_handler.locally_owned_dofs(),
                        DoFTools::extract_locally_relevant_dofs(dof_handler));
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