#
# src/math/matlab/CMakeLists.txt
#
#==============================================================================

# Matlab mex funtion
if (MATLAB_FOUND)
    set(MATLAB_MEXFILE_EXT mexa64)
    include_directories(${MATLAB_INCLUDE_DIR})
    add_library(oskar_random_bpl SHARED oskar_random_bpl.cpp mexFunction.def)
    target_link_libraries(oskar_random_bpl oskar_math ${MATLAB_LIBRARIES})
    set_target_properties(oskar_random_bpl PROPERTIES
        PREFIX "" SUFFIX ".${MATLAB_MEXFILE_EXT}")
endif (MATLAB_FOUND)


# C Wrapper
include_directories(${oskar-lib_SOURCE_DIR}/math)
include_directories(${oskar-lib_SOURCE_DIR}/math/matlab)
add_library(oskar_math_random_bpl SHARED oskar_random_bpl.cpp)
target_link_libraries(oskar_math_random_bpl oskar_math)
