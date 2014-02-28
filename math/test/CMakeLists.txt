#
# src/math/test/CMakeLists.txt
#

set(name math_test)
include_directories(${CUDA_INCLUDE_DIRS})
set(${name}_SRC
    main.cpp
    Test_curand_generate_kernel.cu
    Test_curand.cpp
    Test_find_closest_match.cpp
    Test_linspace.cpp
    Test_Jones.cpp
)
if (OSKAR_USE_LAPACK AND OSKAR_USE_CBLAS)
    list(APPEND ${name}_SRC Test_fit_ellipse.cpp)
endif ()
cuda_add_executable(${name} ${${name}_SRC})
target_link_libraries(${name} oskar gtest ${CUDA_LIBRARIES})
if (OSKAR_USE_LAPACK)
    target_link_libraries(${name} ${OSKAR_LAPACK})
endif()
if (OSKAR_USE_CBLAS)
    target_link_libraries(${name} ${OSKAR_CBLAS})
endif()
set_target_properties(${name} PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}")
add_test(math_test ${name})

