#
# src/math/matlab/CMakeLists.txt
#
#==============================================================================

# Matlab mex funtion
if (MATLAB_FOUND)

    set(MATLAB_MEXFILE_EXT mexa64)

    include_directories(${MATLAB_INCLUDE_DIR})

    add_library(oskar_rbpl SHARED
        oskar_rbpl.cpp
        mexFunction.def)
    target_link_libraries(oskar_rbpl
        oskar_math
        ${MATLAB_LIBRARIES}
    )
    set_target_properties(oskar_rbpl PROPERTIES
        PREFIX ""
        SUFFIX ".${MATLAB_MEXFILE_EXT}"
     )

endif (MATLAB_FOUND)


# C Wrapper
include_directories(${oskar-lib_SOURCE_DIR}/math)
include_directories(${oskar-lib_SOURCE_DIR}/math/matlab)
add_library(oskar_math_rbpl SHARED oskar_math_rbpl.cpp)
target_link_libraries(oskar_math_rbpl oskar_math)

