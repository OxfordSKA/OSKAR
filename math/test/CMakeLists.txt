#
# src/math/test/CMakeLists.txt
#

# ==== CppUnit
if (CPPUNIT_FOUND)

    set(math_test_SRC
        src/CppUnitMain.cpp
        src/CudaInterpBilinearTest.cu
        src/RandomTest.cpp
        src/Matrix3Test.cpp
        src/Matrix4Test.cpp
    )

    include_directories(${CPPUNIT_INCLUDE_DIR})
    cuda_add_executable(mathTest ${math_test_SRC})
    set_target_properties(mathTest PROPERTIES
        COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
        LINK_FLAGS "${OpenMP_CXX_FLAGS}")

    target_link_libraries(mathTest
        oskar
        ${CPPUNIT_LIBRARIES}
    )
endif()



# ==== QtTest
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


