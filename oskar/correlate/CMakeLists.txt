#
# oskar/correlate/CMakeLists.txt
#

set(correlate_SRC
    define_auto_correlate.h
    define_correlate_utils.h
    define_cross_correlate.h
    define_evaluate_auto_power.h
    define_evaluate_cross_power.h
    src/oskar_auto_correlate.c
    src/oskar_cross_correlate_cpu.cl
    src/oskar_cross_correlate_gpu.cl
    src/oskar_correlate_cpu.cl
    src/oskar_correlate_gpu.cl
    src/oskar_correlate.cl
    src/oskar_cross_correlate_omp.cpp
    src/oskar_cross_correlate_scalar_omp.cpp
    src/oskar_cross_correlate.c
    src/oskar_evaluate_auto_power.c
    src/oskar_evaluate_cross_power.c
)

if (CUDA_FOUND)
    list(APPEND correlate_SRC
        src/oskar_correlate.cu
        src/oskar_cross_correlate_cuda.cu
        src/oskar_cross_correlate_scalar_cuda.cu
    )
endif()

set(correlate_SRC "${correlate_SRC}" PARENT_SCOPE)

if (BUILD_TESTING OR NOT DEFINED BUILD_TESTING)
    add_subdirectory(test)
endif()
