#
# src/math/CMakeLists.txt
#

include_directories(${oskar-lib_SOURCE_DIR}/math)

add_subdirectory(matlab)
add_subdirectory(cudak)

# Add the source files in this directory.
set(math_SRC
    src/oskar_mat_tri_c.c
)
foreach (file ${math_cuda_kernels_SRC})
    list(APPEND math_SRC cudak/${file})
endforeach ()

set(math_SRC ${math_SRC}
    src/oskar_cuda_interp_bilinear.cu
)
set(math_SRC "${math_SRC}" PARENT_SCOPE)

add_subdirectory(test)
