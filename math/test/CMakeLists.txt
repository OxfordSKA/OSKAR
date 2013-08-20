#
# src/math/test/CMakeLists.txt
#

# CppUnit
if (CPPUNIT_FOUND)

    set(math_test_SRC
        src/CppUnitMain.cpp
        src/Test_curand.cu
        src/Test_matrix_math.cpp
        src/Test_rotate.cpp
        src/Test_sph_lm.cpp
    )
    if (OSKAR_USE_LAPACK AND OSKAR_USE_CBLAS)
        list(APPEND math_test_SRC src/Test_fit_ellipse.cpp)
    endif ()

    include_directories(${CPPUNIT_INCLUDE_DIR})
    include_directories(${CUDA_INCLUDE_DIRS})
    list(APPEND CUDA_NVCC_FLAGS --compiler-options;${OpenMP_CXX_FLAGS};)
    cuda_add_executable(math_test ${math_test_SRC})
    target_link_libraries(math_test oskar
        ${CPPUNIT_LIBRARIES} ${CUDA_LIBRARIES})
    if (OSKAR_USE_LAPACK)
        target_link_libraries(math_test ${OSKAR_LAPACK})
    endif()
    if (OSKAR_USE_CBLAS)
        target_link_libraries(math_test ${OSKAR_CBLAS})
    endif()
    set_target_properties(math_test PROPERTIES
        COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
        LINK_FLAGS "${OpenMP_CXX_FLAGS}")
    
    add_test(math_test math_test)
endif()

# New test using gtest.
set(name math_test2)
include_directories(${PROJECT_SOURCE_DIR}/extern/gtest-1.6.0/include)
set(${name}_SRC
    src/Test_find_closest_match.cpp
    src/Test_linspace.cpp
    src/Test_Jones.cpp
    src/Test_sph2cart.cpp
)
add_executable(${name} ${${name}_SRC})
target_link_libraries(${name} oskar gtest_main)
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

