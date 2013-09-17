#
# src/math/test/CMakeLists.txt
#

set(name math_test)
include_directories(${PROJECT_SOURCE_DIR}/extern/gtest-1.6.0/include)
include_directories(${CUDA_INCLUDE_DIRS})
list(APPEND CUDA_NVCC_FLAGS --compiler-options;${OpenMP_CXX_FLAGS};)
set(${name}_SRC
    main.cpp
    Test_curand.cu
    Test_find_closest_match.cpp
    Test_linspace.cpp
    Test_Jones.cpp
    Test_sph_lm.cpp
    Test_sph2cart.cpp
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

