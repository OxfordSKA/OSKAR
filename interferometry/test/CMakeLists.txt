#
# src/interferometry/test/CMakeLists.txt
#

if (CPPUNIT_FOUND)

    set(interferometry_test_SRC
        src/CppUnitMain.cpp
        src/Test_add_system_noise.cpp
        src/Test_correlator.cu
        src/Test_station_coord_transforms.cpp
        src/Test_TelescopeModel.cpp
        src/Test_Visibilities.cpp
    )

    include_directories(${CPPUNIT_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})
    add_executable(interferometry_test ${interferometry_test_SRC})
    target_link_libraries(interferometry_test oskar
        ${CPPUNIT_LIBRARIES} ${CUDA_LIBRARIES}
    )

    add_test(interferometry_test interferometry_test)

endif()
