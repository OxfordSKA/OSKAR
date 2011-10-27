#
# src/interferometry/test/CMakeLists.txt
#

if (CPPUNIT_FOUND)

    set(interferometry_test_SRC
        src/CppUnitMain.cu
        src/CudaCorrelatorTest.cu
        src/oskar_StationCoordTransformTest.cpp
        src/oskar_Visibilities_Test.cpp
    )

    include_directories(${CPPUNIT_INCLUDE_DIR})
    cuda_add_executable(interferometryTest ${interferometry_test_SRC})
    target_link_libraries(interferometryTest
        oskar
        ${CPPUNIT_LIBRARIES}
    )

    add_test(interferometryTest interferometryTest)

endif()
