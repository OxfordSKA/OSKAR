#
# src/sky/test/CMakeLists.txt
#

if (CPPUNIT_FOUND)

    set(sky_test_SRC
        src/CppUnitMain.cpp
        src/Test_coord_transforms.cpp
        src/Test_SkyModel.cpp
        src/Test_generators.cpp
    )

    include_directories(${CPPUNIT_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})
    add_executable(sky_test ${sky_test_SRC})
    target_link_libraries(sky_test oskar ${CPPUNIT_LIBRARIES} ${CUDA_LIBRARIES})

    add_test(sky_test sky_test)

endif ()
