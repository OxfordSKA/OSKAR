#
# src/math/test/CMakeLists.txt
#

add_subdirectory(cudak)

# CppUnit
if (CPPUNIT_FOUND)

    set(math_test_SRC
        src/CppUnitMain.cpp
        src/Test_curand.cu
        src/Test_healpix.cpp
        src/Test_Jones.cpp
        src/Test_Matrix3.cpp
        src/Test_Matrix4.cpp
        src/Test_meshgrid.cpp
        src/Test_linspace.cpp
        src/Test_matrix_math.cpp
        src/Test_rotate.cpp
        src/Test_sph_lm.cpp
        src/Test_sph2cart.cpp
        src/Test_SphericalPositions.cpp
    )
    if (OSKAR_USE_LAPACK AND OSKAR_USE_CBLAS)
        list(APPEND math_test_SRC src/Test_fit_ellipse.cpp)
    endif ()

    include_directories(${CPPUNIT_INCLUDE_DIR})
    include_directories(${CUDA_INCLUDE_DIRS})
    list(APPEND CUDA_NVCC_FLAGS --compiler-options;${OpenMP_CXX_FLAGS};)
    cuda_add_executable(math_test ${math_test_SRC})
    target_link_libraries(math_test oskar oskar_math_test_cudak
        ${CPPUNIT_LIBRARIES} ${CUDA_LIBRARIES})
    if (OSKAR_USE_LAPACK)
        target_link_libraries(math_test ${OSKAR_LAPACK})
    endif()
    if (OSKAR_USE_CBLAS)
        target_link_libraries(math_test ${OSKAR_CBLAS})
    endif()

    add_test(math_test math_test)
endif()
