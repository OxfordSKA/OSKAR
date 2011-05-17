#
# src/modules/CMakeLists.txt
#
#==============================================================================
include_directories(
    ${oskar-lib_SOURCE_DIR}/modules
    ${oskar-lib_SOURCE_DIR}/math
)
add_subdirectory(cuda)

# === Get list of source files required to build the math module.
set(modules_SRC)
foreach (file ${modules_cuda_SRC})
    list(APPEND modules_SRC cuda/${file})
endforeach ()
#foreach (file ${modules_sky_SRC})
#    list(APPEND modules_SRC sky/${file})
#endforeach ()

# === Macros to build the oskar-lib math module.
cuda_add_library(oskar_modules ${modules_SRC})
target_link_libraries(oskar_modules
    ${CUDA_CUBLAS_LIBRARIES}
    oskar_cuda
    oskar_math
)
set_target_properties(oskar_modules PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}"
)


# === Install configuration for the oskar-lib cuda module.
install(TARGETS oskar_modules DESTINATION ${OSKAR_LIB_INSTALL_DIR})
set_target_properties(oskar_modules PROPERTIES
    SOVERSION ${oskar-lib_SOVERSION})
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DESTINATION ${OSKAR_INCLUDE_INSTALL_DIR}
    FILES_MATCHING PATTERN "*.h"
    PATTERN "src" EXCLUDE
    PATTERN ".svn" EXCLUDE
    PATTERN "test" EXCLUDE)
