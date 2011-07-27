#
# src/math/CMakeLists.txt
#
#==============================================================================

include_directories(${oskar-lib_SOURCE_DIR}/math)

# Add the source files in this directory.
set(math_SRC
    src/oskar_math_mat_tri_c.c
)


add_subdirectory(cudak)
foreach (file ${math_cuda_kernels_SRC})
    list(APPEND math_SRC cudak/${file})
endforeach ()
set(math_SRC ${math_SRC}
    src/oskar_math_cuda_interp_bilinear.cu
)
set(math_SRC "${math_SRC}" PARENT_SCOPE)


# === Macros to build the oskar-lib math module.
cuda_add_library(oskar_math ${math_SRC})
set_target_properties(oskar_math PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}")
target_link_libraries(oskar_math
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDA_LIBRARIES})
set_target_properties(oskar_math PROPERTIES
    INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib
    INSTALL_RPATH_USE_LINK_PATH TRUE)


# === Install target for the oskar-lib math library module.
if (NOT WIN32)
    install(TARGETS oskar_math LIBRARY DESTINATION ${OSKAR_LIB_INSTALL_DIR})
    set_target_properties(oskar_math PROPERTIES
        SOVERSION ${oskar-lib_SOVERSION})
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DESTINATION ${OSKAR_INCLUDE_INSTALL_DIR}
        FILES_MATCHING PATTERN "*.h"
        PATTERN "src" EXCLUDE
        PATTERN "test" EXCLUDE
        PATTERN ".svn" EXCLUDE)
endif()

# === Recurse into test directory.
add_subdirectory(test)

add_subdirectory(matlab)
