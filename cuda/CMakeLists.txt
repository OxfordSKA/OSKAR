#
# src/cuda/CMakeLists.txt
#
#==============================================================================

# Only build this module if CUDA is found.
if (CUDA_FOUND)

    # === Recurse into the kernels sub-directory.
    add_subdirectory(kernels)

    # === Create list of source files for the oskar-lib cuda module.
    set(cuda_SRC
        # Callable functions:
        src/oskar_cuda_as2hi.cu
        src/oskar_cuda_bf2hig.cu
        src/oskar_cuda_bfmv.cu
        src/oskar_cuda_bp2hcgg.cu
        src/oskar_cuda_bp2hcggu.cu
        src/oskar_cuda_bp2hig.cu
        src/oskar_cuda_bp2higu.cu
        src/oskar_cuda_bp2hugg.cu
        src/oskar_cuda_bp2huggu.cu
        src/oskar_cuda_eq2hg.cu
        src/oskar_cuda_hbp2hig.cu
        src/oskar_cuda_im2dft.cu
        src/oskar_cuda_im2dftlm.cu
        src/oskar_cuda_le2hg.cu
        src/oskar_cuda_rpw3leg.cu
        src/oskar_cuda_vecadd.cu
    )
    foreach (file ${cuda_kernels_SRC})
        list(APPEND cuda_SRC kernels/${file})
    endforeach ()

    # === Macros to build the oskar-lib cuda module.
    set(CUDA_GENERATED_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/oskar_cuda.dir/src)
    include_directories(${CUDA_INCLUDE_DIRS})
    cuda_add_library(oskar_cuda ${cuda_SRC})
    target_link_libraries(oskar_cuda ${CUDA_CUBLAS_LIBRARIES})

    # === Install configuration for the oskar-lib cuda module.
    install(TARGETS oskar_cuda LIBRARY DESTINATION ${OSKAR_LIB_INSTALL_DIR})
    set_target_properties(oskar_cuda PROPERTIES
        SOVERSION ${oskar-lib_SOVERSION})
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DESTINATION ${OSKAR_INCLUDE_INSTALL_DIR}
        FILES_MATCHING PATTERN "*.h"
        PATTERN "src" EXCLUDE
        PATTERN ".svn" EXCLUDE
        PATTERN "kernels" EXCLUDE
        PATTERN "test" EXCLUDE)

    # === Add a target for clean of nvcc generated files.
    cuda_build_clean_target()

    # === Recurse into test subdirectory.
    add_subdirectory(test)

endif (CUDA_FOUND)
