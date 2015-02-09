#
# src/correlate/CMakeLists.txt
#

# Add the source files in this directory.
set(correlate_SRC
    src/oskar_correlate_gaussian_omp.c
    src/oskar_correlate_gaussian_time_smearing_omp.c
    src/oskar_correlate_point_omp.c
    src/oskar_correlate_point_time_smearing_omp.c
    src/oskar_correlate_scalar_gaussian_omp.c
    src/oskar_correlate_scalar_gaussian_time_smearing_omp.c
    src/oskar_correlate_scalar_point_omp.c
    src/oskar_correlate_scalar_point_time_smearing_omp.c
    src/oskar_correlate.c
    src/oskar_evaluate_average_cross_power_beam.c
    src/oskar_evaluate_average_cross_power_beam_omp.c
    src/oskar_evaluate_average_scalar_cross_power_beam_omp.c
)

if (CUDA_FOUND)
    list(APPEND correlate_SRC
        src/oskar_correlate_gaussian_cuda.cu
        src/oskar_correlate_gaussian_time_smearing_cuda.cu
        src/oskar_correlate_point_cuda.cu
        src/oskar_correlate_point_time_smearing_cuda.cu
        src/oskar_correlate_point_time_smearing_new_cuda.cu
        src/oskar_correlate_scalar_gaussian_cuda.cu
        src/oskar_correlate_scalar_gaussian_time_smearing_cuda.cu
        src/oskar_correlate_scalar_point_cuda.cu
        src/oskar_correlate_scalar_point_time_smearing_cuda.cu
        src/oskar_evaluate_average_cross_power_beam_cuda.cu
        src/oskar_evaluate_average_scalar_cross_power_beam_cuda.cu
    )
endif()

set(correlate_SRC "${correlate_SRC}" PARENT_SCOPE)

add_subdirectory(test)
