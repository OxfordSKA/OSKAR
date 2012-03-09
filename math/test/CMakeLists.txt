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
        src/Test_linspace.cpp
        src/Test_matrix_math.cpp
        src/Test_meshgrid.cpp
        src/Test_rotate.cpp
        src/Test_sph_lm.cpp
        src/Test_sph2cart.cpp
        src/Test_SphericalPositions.cpp
    )

    include_directories(${CPPUNIT_INCLUDE_DIR})
    include_directories(${CUDA_INCLUDE_DIRS})
    list(APPEND CUDA_NVCC_FLAGS --compiler-options;${OpenMP_CXX_FLAGS};)
    cuda_add_executable(math_test ${math_test_SRC})
    target_link_libraries(math_test oskar oskar_math_test_cudak
        ${CPPUNIT_LIBRARIES} ${CUDA_LIBRARIES})
    add_test(math_test math_test)
endif()


# QtTest
if (QT_QTTEST_FOUND AND BUILD_OSKAR_WIDGETS)

    include_directories(
        ${Qwt5_INCLUDE_DIR}
        ${QT_INCLUDE_DIR}
        ${QT_QTCORE_INCLUDE_DIR}
        ${QT_QTGUI_INCLUDE_DIR}
    )

    qt4_wrap_cpp(QTest_Random_MOC QTest_Random.h)
    add_executable(qtest_Random ${QTest_Random_MOC})
    target_link_libraries(qtest_Random
        oskar
        oskar_widgets
        ${QT_QTTEST_LIBRARY}
        ${QT_QTGUI_LIBRARY}
    )
endif()
