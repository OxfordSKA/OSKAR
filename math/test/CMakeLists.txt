#
# src/math/test
#

# CppUnit
# ==============================================================================
if (CPPUNIT_FOUND)

    set(math_test_SRC
        src/CppUnitMain.cpp
        src/RandomTest.cpp
    )

    if (CUDA_FOUND)
        set(math_test_SRC ${math_test_SRC}
            src/CudaInterpBilinearTest.cu
        )
    endif()

    include_directories(${CPPUNIT_INCLUDE_DIR})
    if (CUDA_FOUND)
        cuda_add_executable(mathTest ${math_test_SRC})
    else()
        add_executable(mathTest ${math_test_SRC})
    endif()
    set_target_properties(mathTest PROPERTIES
        COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
        #LINK_FLAGS "${OpenMP_CXX_FLAGS}"
    )
    target_link_libraries(mathTest
        oskar_math
        ${CPPUNIT_LIBRARIES}
    )
endif()


# === QtTest unit testing.
if (QT_QTTEST_FOUND)

    include_directories(
        ${Qwt5_INCLUDE_DIR}
        ${QT_INCLUDE_DIR}
        ${QT_QTCORE_INCLUDE_DIR}
        ${QT_QTGUI_INCLUDE_DIR}
    )

    # Random test.
    qt4_wrap_cpp(QTest_Random_MOC
        QTest_Random.h
    )
    add_executable(qtest_Random ${QTest_Random_MOC})
    target_link_libraries(qtest_Random
        oskar_math
        oskar_widgets_plotting
        ${QT_QTTEST_LIBRARY}
        ${QT_QTGUI_LIBRARY}
    )
endif()


