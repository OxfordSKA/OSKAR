#
# src/modules/cuda/CMakeLists.txt
#
#==============================================================================

# Only build this module if CUDA is found.
if (CUDA_FOUND)

    # === Create list of source files for the oskar-lib cuda module.
    set(modules_cuda_SRC
        src/oskar_modules_cuda_correlator_lm.cu
        src/oskar_modules_cuda_correlator_lm_bw.cu
    )
    set(modules_cuda_SRC "${modules_cuda_SRC}" PARENT_SCOPE)

    # === Macros to build the oskar-lib cuda module.
    set(CUDA_GENERATED_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/oskar_modules_cuda.dir/src)

    cuda_add_library(oskar_modules_cuda SHARED ${modules_cuda_SRC})
    set_target_properties(oskar_modules_cuda PROPERTIES PREFIX "lib")
    set_target_properties(oskar_modules_cuda PROPERTIES
        COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    )
    target_link_libraries(oskar_modules_cuda
        ${CUDA_CUBLAS_LIBRARIES}
        oskar_cuda
        oskar_math
    )

    # === Install configuration for the oskar-lib cuda module.
    install(TARGETS oskar_modules_cuda DESTINATION ${OSKAR_LIB_INSTALL_DIR})
    set_target_properties(oskar_modules_cuda PROPERTIES
        SOVERSION ${oskar-lib_SOVERSION})
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DESTINATION ${OSKAR_INCLUDE_INSTALL_DIR}
        FILES_MATCHING PATTERN "*.h"
        PATTERN "src" EXCLUDE
        PATTERN ".svn" EXCLUDE
        PATTERN "test" EXCLUDE)

    # === Add a target for clean of nvcc generated files.
    #cuda_build_clean_target()

    # === Recurse into test subdirectory.
    add_subdirectory(test)

endif (CUDA_FOUND)
