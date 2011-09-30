#
# src/sky/test/CMakeLists.txt
#

# CppUnit tests.
if (CPPUNIT_FOUND)

    set(sky_test_SRC
        src/CppUnitMain.cpp
        src/SkyTest.cpp
        src/EquatorialToHorizontalTest.cu
        src/SkyModelTest.cpp
    )

    cuda_add_executable(skyTest ${sky_test_SRC})
    target_link_libraries(skyTest oskar ${CPPUNIT_LIBRARIES})

    add_test(skyTest skyTest)

endif ()
