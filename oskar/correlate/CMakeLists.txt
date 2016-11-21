#
# oskar/correlate/CMakeLists.txt
#

set(correlate_SRC
    src/oskar_auto_correlate.c
    src/oskar_auto_correlate_omp.c
    src/oskar_auto_correlate_scalar_omp.c
    src/oskar_cross_correlate_gaussian_omp.c
    src/oskar_cross_correlate_gaussian_time_smearing_omp.c
    src/oskar_cross_correlate_point_omp.c
    src/oskar_cross_correlate_point_time_smearing_omp.c
    src/oskar_cross_correlate_gaussian_scalar_omp.c
    src/oskar_cross_correlate_gaussian_time_smearing_scalar_omp.c
    src/oskar_cross_correlate_point_scalar_omp.c
    src/oskar_cross_correlate_point_time_smearing_scalar_omp.c
    src/oskar_cross_correlate.c
    src/oskar_evaluate_auto_power.c
    src/oskar_evaluate_auto_power_c.c
    src/oskar_evaluate_cross_power.c
    src/oskar_evaluate_cross_power_omp.c)

if (CUDA_FOUND)
    list(APPEND correlate_SRC
        src/oskar_auto_correlate_cuda.cu
        src/oskar_auto_correlate_scalar_cuda.cu
        src/oskar_cross_correlate_gaussian_cuda.cu
        src/oskar_cross_correlate_gaussian_time_smearing_cuda.cu
        src/oskar_cross_correlate_point_cuda.cu
        src/oskar_cross_correlate_point_time_smearing_cuda.cu
        src/oskar_cross_correlate_gaussian_scalar_cuda.cu
        src/oskar_cross_correlate_gaussian_time_smearing_scalar_cuda.cu
        src/oskar_cross_correlate_point_scalar_cuda.cu
        src/oskar_cross_correlate_point_time_smearing_scalar_cuda.cu
        src/oskar_evaluate_auto_power_cuda.cu
        src/oskar_evaluate_cross_power_cuda.cu)
endif()

set(correlate_SRC "${correlate_SRC}" PARENT_SCOPE)

add_subdirectory(test)
