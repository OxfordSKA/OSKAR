#
# src/interferometry/test/CMakeLists.txt
#

if (CPPUNIT_FOUND)

    set(interferometry_cuda_test_SRC
        src/CppUnitMain.cpp
        src/ModuleCudaCorrelatorTest.cpp
    )

    include_directories(${CPPUNIT_INCLUDE_DIR})
    add_executable(interferometryCudaTest ${interferometry_cuda_test_SRC})
    target_link_libraries(interferometryCudaTest
        oskar
        ${CPPUNIT_LIBRARIES}
    )

endif()
