#
# src/math/test/CMakeLists.txt
#

# ==== CppUnit
if (CPPUNIT_FOUND)

    set(math_test_SRC
        src/CppUnitMain.cpp
        src/Test_healpix.cpp
        src/Test_Jones.cpp
        src/Test_Matrix3.cpp
        src/Test_Matrix4.cpp
        src/Test_linspace.cpp
        src/Test_meshgrid.cpp
        src/Test_SphericalPositions.cpp
    )

    include_directories(${CPPUNIT_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})
    add_executable(math_test ${math_test_SRC})
    target_link_libraries(math_test oskar ${CPPUNIT_LIBRARIES} ${CUDA_LIBRARIES})

    add_test(math_test math_test)
endif()


# FIXME Move to CppUnit test.
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
