#include "MatrixFreeSolver.hpp"

using namespace dealii;
template <int dim, int fe_degree, std::floating_point NumberType>
  MatrixFreeSolver<dim, fe_degree, NumberType>::MatrixFreeSolver()
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
    , setup_time(0.0)
    , pcout(std::cout,
            (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0))
    , time_details(std::cout,
                   (false && Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)) //remove false to enable time details (do we want it?)
  {}